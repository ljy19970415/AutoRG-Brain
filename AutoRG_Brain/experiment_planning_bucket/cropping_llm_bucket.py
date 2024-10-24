#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict

from petrel_client.client import Client
from petrel_client.version import version
import os


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def get_case_identifier(case):
    # case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    # if case[-1].endswith('_ab_mask.nii.gz'):
    #     case_identifier = case[-1].split("/")[-1].split(".nii.gz")[0][:-8]
    # elif case[-1].endswith('_ana_mask.nii.gz'):
    #     case_identifier = case[-1].split("/")[-1].split(".nii.gz")[0][:-9]
    # elif case[-1].endswith('_bbox.nii.gz'):
    #     case_identifier = case[-1].split("/")[-1].split(".nii.gz")[0][:-5]
    # else:
    #     case_identifier = case[-1].split("/")[-1].split(".nii.gz")[0]
    if 'WMH_Segmentation_Challenge' in case[0]:
        case_identifier = case[0].split('/')[-4]+'_'+case[0].split('/')[-3]+'_'+case[0].split('/')[-1].split('.')[0]
    elif '6th_normal_mask' in case[0]:
        case_identifier = case[0].split('/')[-2]+'_'+case[0].split('/')[-1].split('.')[0]
    elif 'myDWI' in case[0]:
        case_identifier = case[0].split('/')[-2]+'_'+case[0].split('/')[-1].split('.')[0]
    else:
        case_identifier = case[0].split('/')[-1].split('.')[0]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file1=None, seg_file2=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file1"] = seg_file1
    properties["seg_file2"] = seg_file2

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    # 1 ,512, 512, 146
    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])

    if seg_file1 is not None and seg_file2 is not None:
        seg_itk1 = sitk.ReadImage(seg_file1)
        seg_itk2 = sitk.ReadImage(seg_file2)
        seg_npy = np.vstack([sitk.GetArrayFromImage(d)[None].astype(np.float32) for d in [seg_itk1, seg_itk2]])
    else:
        seg_npy = None

    return data_npy.astype(np.float32), seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox

def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None, bucket_output_folder=None, bucket_preprocess_output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads
        self.bucket_output_folder = bucket_output_folder

        self.bucket_preprocess_output_folder = bucket_preprocess_output_folder

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg[0]) # seg[0] means the value of key "label1", here is the ..ana_mask.nii.gz
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file1=None, seg_file2=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file1, seg_file2)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            # if overwrite_existing \
            #         or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
            #             or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

            data, seg, properties = self.crop_from_list_of_files(case[:1], case[1], case[-1])

            # seg shape 2 x,y,z

            all_data = np.vstack((data, seg))

            npz_file_path = os.path.join(self.output_folder, "%s.npz" % case_identifier)
            np.savez_compressed(npz_file_path, data=all_data)
            
            ### Note that If you don't want to save the local file to s3 bucket
            ### Just delete the following three lines ##

            save_aws_image_path = self.bucket_output_folder+'//'+npz_file_path.split('/')[-1]
            ## move the file from local to aws s3 bucket storage
            os.system(f'aws s3 cp {npz_file_path} {save_aws_image_path}  --endpoint-url=http://10.140.14.204')
            ## delete the local file afterwards
            os.remove(npz_file_path)

            with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder
        
        # save the ana mask in gt_segmentation folder
        output_folder_gt = self.bucket_preprocess_output_folder+"//gt_segmentations"
        # maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            if case[1] is not None:
                save_image_path = case[1]
                ### Note that if you don't want to save the image to bucket
                ## Just delete the following two lines
                save_aws_image_path = output_folder_gt+'//'+save_image_path.split('/')[-1]
                os.system(f'aws s3 cp {save_image_path} {save_aws_image_path}  --endpoint-url=http://10.140.14.204')
                # shutil.copy(case[1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))
            # case: [img_path, label_path1, label_path2]
            # case_identifier: a10923272
        # list_of_args [[[img_path,label_path],'a102323231',False],..]
        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
