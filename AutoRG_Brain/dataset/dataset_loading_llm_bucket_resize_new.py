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

from collections import OrderedDict
import numpy as np
from multiprocessing import Pool

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from configuration import default_num_threads
from paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *

from .utils import SynthesisTumor as SynthesisTumor_intense
# from .copypaste import SynthesisTumor as SynthesisTumor_copypaste
import SimpleITK as sitk

import random
import json

from io import BytesIO
import os

from .utils import nnUNet_resize

def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_case_identifiers_from_raw_folder(folder):
    case_identifiers = np.unique(
        [i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        print("unpack",npz_file.split('/')[-1])
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)

def load_from_bucket(aws_path, client=None):

    data = client.get(aws_path)
    value_buf = memoryview(data)
    iostr = BytesIO(value_buf)
    img_array = np.load(iostr)
    return img_array

def convert_to_npy_bucket(args):

    npz_file, npz_file_bucket, key, client = args
    
    npy_bucket_path = npz_file_bucket[:-3] + "npy"
    if not client.contains(npy_bucket_path):
        print("unpack",npz_file_bucket)
        a = load_from_bucket(npz_file_bucket, client=client)[key]
        save_local_path = npz_file[:-3] + "npy"
        print("save",save_local_path)
        np.save(save_local_path, a)
        os.system(f'aws s3 cp {save_local_path} {npy_bucket_path}  --endpoint-url=http://10.140.14.204')
        if os.path.exists(save_local_path):
            print("remove",save_local_path)
            os.remove(save_local_path)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset_bucket(folder, folder_bucket, threads=default_num_threads, key="data",train_file = None, client=None):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)

    npz_files = list(map(lambda x:join(folder,x+'.npz'), train_file['training'])) + list(map(lambda x:join(folder,x+'.npz'), train_file['validation']))

    npz_files_bucket = list(map(lambda x:folder_bucket+'//'+x+'.npz', train_file['training'])) + list(map(lambda x:folder_bucket+'//'+x+'.npz', train_file['validation']))
    
    p.map(convert_to_npy_bucket, zip(npz_files, npz_files_bucket, [key] * len(npz_files),[client] * len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key] * len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset_bucket(folder, folder_bucket, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading dataset')
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("pkl")]
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = folder_bucket + "//" + "%s.npz" % c

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = folder_bucket + "//" + "%s_segs.npz" % c

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return:
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result


class DataLoader3D_bucket(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, report=None, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None,client=None,dataset="six"):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D_bucket, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        # list_of_keys = list(self._data.keys()) # self._data = data
        self.list_of_keys = list(self._data.keys()) # self._data = data
        # if dataset == "six":
        
        self.mode = dataset

        # case_dic = json.load(open('/mnt/petrelfs/leijiayu/nnUNet/nnUNet_raw/nnUNet_raw_data/Task005_llm_full/case_dic.json','r'))
        case_dic = json.load(open('raw_data/Task002_llm_test/case_dic.json','r'))
        self.list_of_keys_modal = {'DWI':list(filter(lambda x:x in case_dic['DWI'],self.list_of_keys)),'T1WI':list(filter(lambda x:x in case_dic['T1WI'],self.list_of_keys)),'T2WI':list(filter(lambda x:x in case_dic['T2WI'], self.list_of_keys)),'T2FLAIR':list(filter(lambda x:x in case_dic['T2FLAIR'], self.list_of_keys))}
        del case_dic
        
        # case_dic_FT = json.load(open('/mnt/petrelfs/leijiayu/nnUNet/nnUNet_raw/nnUNet_raw_data/Task010_llm_FT/case_dic.json','r'))
        # case_dic_FT = json.load(open('/mnt/petrelfs/leijiayu/nnUNet/nnUNet_raw/nnUNet_raw_data/Task021_llm_test/case_dic.json','r'))
        # self.list_of_keys_modal_FT = {'DWI':list(filter(lambda x:x in case_dic_FT['DWI'],self.list_of_keys)),'T1WI':list(filter(lambda x:x in case_dic_FT['T1WI'],self.list_of_keys)),'T2WI':list(filter(lambda x:x in case_dic_FT['T2WI'], self.list_of_keys)),'T2FLAIR':list(filter(lambda x:x in case_dic_FT['T2FLAIR'], self.list_of_keys))}
        # del case_dic_FT

        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None: # pad_sides is None
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides

        self.client = client
        
        self.batch_size = batch_size

        print("222 slef.batch_size",self.batch_size)

        self.data_shape, self.seg_shape = self.determine_shapes()

        self.report = report

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))


    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else: # choose this
            num_seg = 1

        k = list(self._data.keys())[0]

        if self.client.contains(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = load_from_bucket(self._data[k]['data_file'][:-4] + ".npy", client=self.client)
        else:
            case_all_data = load_from_bucket(self._data[k]['data_file'],client=self.client)['data']
        
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, 1, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)

        return data_shape, seg_shape
    
    # self._data[i] = {
    # "data_file": join(self.folder_with_preprocessed_data,"%s.npz" % i),
    # "properties_file": join(self.folder_with_preprocessed_data,"%s.pkl" % i),
    # "properties": load_pickle(self.dataset[i]["properties_file"]) 
    # }

    def generate_train_batch(self):

        # choose a modal
        choose_modal = np.random.choice(['DWI','T1WI','T2WI','T2FLAIR'], p=[0.25,0.25,0.25,0.25])

        ### old ###
        # if self.mode=='FT':
        #     half_batch = self.batch_size // 2
        #     selected_keys1 = np.random.choice(self.list_of_keys_modal[choose_modal], half_batch, False, None) 
        #     selected_keys2 = np.random.choice(self.list_of_keys_modal_FT[choose_modal], half_batch, False, None)
        #     selected_keys = np.concatenate((selected_keys1,selected_keys2))
        #     random.shuffle(selected_keys)
        # elif self.mode == 'public':
        #     selected_keys = np.random.choice(self.list_of_keys_modal_FT[choose_modal], self.batch_size, False, None)
        # else:
        #     selected_keys = np.random.choice(self.list_of_keys_modal[choose_modal], self.batch_size, False, None) # pick batch_size samples，samples may repeat
        ### new ###
        selected_keys = np.random.choice(self.list_of_keys_modal[choose_modal], self.batch_size, False, None) # pick batch_size samples，samples may repeat

        modal = choose_modal

        data = np.zeros(self.data_shape, dtype=np.float32) # b, c, patch_size
        seg = np.zeros(self.seg_shape, dtype=np.float32) # b, 1, patch_size
        case_properties = []

        reports = []

        for j, i in enumerate(selected_keys):

            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j): # 0.33 probabilities to force_fg
                force_fg = True
            else:
                force_fg = False

            reports.append(self.report[i]) # self.report is self.report["region_report"]["training/validation"]

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            
            if self.client.contains(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data_origin = load_from_bucket(self._data[i]['data_file'][:-4] + ".npy", client=self.client)
            else:
                case_all_data_origin = load_from_bucket(self._data[i]['data_file'], client=self.client)['data']
            
            # data: case_all_data[0].shape = (original_x, original_y, original_z)
            # seg: case_all_data[1].shape = (original_x, original_y, original_z)

            case_all_data = case_all_data_origin.copy()

            data[j] = np.expand_dims(nnUNet_resize(case_all_data[0,:],self.final_patch_size,axis=0),axis=0)
            seg[j,0] = np.expand_dims(nnUNet_resize(case_all_data[1,:],self.final_patch_size,is_seg=True,axis=0),axis=0)
            seg[j,1] = np.expand_dims(nnUNet_resize(case_all_data[2,:],self.final_patch_size,is_seg=True,axis=0),axis=0)

        return {'data': data, 'seg': seg, 'modal':modal, "report":reports}

if __name__ == "__main__":
    t = "Task002_Heart"
    p = join(preprocessing_output_dir, t, "stage1")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "plans_stage1.pkl"), 'rb') as f:
        plans = pickle.load(f)
    unpack_dataset(p)
    dl = DataLoader3D(dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33)
    dl = DataLoader3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2,
                      oversample_foreground_percent=0.33)
    dl2d = DataLoader2D(dataset, (64, 64), np.array(plans['patch_size']).astype(int)[1:], 12,
                        oversample_foreground_percent=0.33)
