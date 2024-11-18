import nibabel as nib
import numpy as np
import os

from scipy.ndimage import binary_fill_holes
import cv2
import random
import SimpleITK as sitk
# from skimage.transform import resize
# from scipy.ndimage.interpolation import map_coordinates
# from collections import OrderedDict

# from .batchgenerator import *

from .batchgenerator import *

from augmentation.vtk_itk import pd_to_itk_image
from augmentation.prelin_sphere import *

import vtk
import torchio as tio

from scipy.ndimage import gaussian_filter

import json

# def nib_load(file_name,component=0):
#     if not os.path.exists(file_name):
#         print('Invalid file name, can not find the file!')
#     proxy = nib.load(file_name)
#     data = proxy.get_fdata()
#     if data.ndim>3:
#         data=data[:,:,:,component]
#     proxy.uncache()
#     return data


ployhedron_path = 'AutoRG_Brain/augmentation/resources/geodesic_polyhedron.vtp'
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(ployhedron_path)
reader.Update()
input_poly_data = reader.GetOutput()

trans =tio.Compose((
        tio.RandomElasticDeformation(num_control_points=7, locked_borders=2),
        tio.RandomBlur()))

# def nnUNet_resize(data, new_shape, do_separate_z=True, is_seg=False, axis=2, order=3, order_z=0):
#     assert len(data.shape) == 3, "data must be (x, y, z)"
#     assert len(new_shape) == len(data.shape)

#     if is_seg:
#         resize_fn = resize_segmentation
#         kwargs = OrderedDict()
#         order = 1
#     else:
#         resize_fn = resize
#     kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
#     dtype_data = data.dtype
#     shape = np.array(data.shape)
#     new_shape = np.array(new_shape)
#     if np.any(shape != new_shape):
#         data = data.astype(float)
#         if do_separate_z:
#             #print("separate z, order in z is", order_z, "order inplane is", order)
#             if axis == 0:
#                 new_shape_2d = new_shape[1:]
#             elif axis == 1:
#                 new_shape_2d = new_shape[[0, 2]]
#             else:
#                 new_shape_2d = new_shape[:-1]

#             reshaped_final_data = []

#             reshaped_data = []
#             for slice_id in range(shape[axis]):
#                 if axis == 0:
#                     reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
#                 elif axis == 1:
#                     reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
#                 else:
#                     reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
#             reshaped_data = np.stack(reshaped_data, axis)
#             # print("reshaped_data",reshaped_data.shape)
#             if shape[axis] != new_shape[axis]:

#                 # The following few lines are blatantly copied and modified from sklearn's resize()
#                 rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
#                 orig_rows, orig_cols, orig_dim = reshaped_data.shape

#                 row_scale = float(orig_rows) / rows
#                 col_scale = float(orig_cols) / cols
#                 dim_scale = float(orig_dim) / dim

#                 map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
#                 map_rows = row_scale * (map_rows + 0.5) - 0.5
#                 map_cols = col_scale * (map_cols + 0.5) - 0.5
#                 map_dims = dim_scale * (map_dims + 0.5) - 0.5

#                 coord_map = np.array([map_rows, map_cols, map_dims])
#                 reshaped_data = map_coordinates(reshaped_data, coord_map, order=order_z, mode='nearest').astype(dtype_data)
                
#                 #print("shape[axis] != new_shape[axis]",reshaped_data.shape)
#                 # else:
#                 #     unique_labels = np.unique(reshaped_data)
#                 #     reshaped = np.zeros(new_shape, dtype=dtype_data)

#                 #     for i, cl in enumerate(unique_labels):
#                 #         reshaped_multihot = np.round(
#                 #             map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
#                 #                             mode='nearest'))
#                 #         reshaped[reshaped_multihot > 0.5] = cl
#                 #     reshaped_final_data.append(reshaped[None].astype(dtype_data))
#             # else:
#             #     reshaped_data = reshaped_data[None].astype(dtype_data)
#         else:
#             reshaped_data = resize_fn(data, new_shape, order, **kwargs).astype(dtype_data)
#             # print("no separate z, order", reshaped_data.shape)
#         return reshaped_data.astype(dtype_data)
#     else:
#         print("no resampling necessary",data.shape)
#         return data

# def resize_segmentation(segmentation, new_shape, order=3, cval=0):
#     '''
#     Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
#     hot encoding which is resized and transformed back to a segmentation map.
#     This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
#     :param segmentation:
#     :param new_shape:
#     :param order:
#     :return:
#     '''
#     tpe = segmentation.dtype
#     unique_labels = np.unique(segmentation)
#     assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
#     if order == 0:
#         return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
#     else:
#         reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

#         for i, c in enumerate(unique_labels):
#             mask = segmentation == c
#             reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
#             reshaped[reshaped_multihot >= 0.5] = c
#         return reshaped

def tumor_array_generate(img_array, itk_img, input_poly_data, name):
    brain_start_x,brain_end_x,brain_start_y,brain_end_y,brain_start_z,brain_end_z = crop(img_array)
    ratio_x = np.random.uniform(0.3,0.8)
    ratio_y = np.random.uniform(0.2,0.7)
    if np.random.random()< 0.5:
        ratio_z = np.random.uniform(0.24,0.44)
    else:
        ratio_z = np.random.uniform(0.56,0.76)
    
    tumor_center_x = int((brain_end_x-brain_start_x)*ratio_x + brain_start_x)
    tumor_center_y = int((brain_end_y-brain_start_y)*ratio_y + brain_start_y)
    tumor_center_z = int((brain_end_z-brain_start_z)*ratio_z + brain_start_z)
    center = [tumor_center_z,tumor_center_y,tumor_center_x]

    center = [tumor_center_x,tumor_center_y,tumor_center_z]

    center = [128,128,128]

    k = 4/3 * np.pi
    volume = np.random.randint(7000, 200000)
    radius = (volume / k) ** (1/3)
    ratio = np.random.uniform(0.8, 1)
    a = radius
    b = radius / ratio
    c = radius * ratio

    radii = c,b,a
    angles = np.random.uniform(0, 180, size=3)

    octaves = np.random.randint(4, 8)
    offset = np.random.randint(1000)
    scale = 0.5

    output_poly_data = get_resection_poly_data(input_poly_data,offset,[64,64,64],radii,angles,octaves,scale)

    output_poly_stk, ndseg = pd_to_itk_image(output_poly_data,itk_img)

    ### for test ###      

    output_poly_array = sitk.GetArrayFromImage(output_poly_stk)
    output_poly_array[output_poly_array>0]=1
    
    mask = np.zeros(img_array.shape)
    x_start, x_end = np.where(np.any(output_poly_array!=0, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(output_poly_array!=0, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(output_poly_array!=0, axis=(0, 1)))[0][[0, -1]]
    output_poly_array = output_poly_array[x_start:x_end,y_start:y_end,z_start:z_end]

    cx,cy,cz = tumor_center_x,tumor_center_y,tumor_center_z
    axl,ayl,azl = output_poly_array.shape[0], output_poly_array.shape[1], output_poly_array.shape[2]
    mask[max(0,cx-axl//2):cx+axl-axl//2, max(0,cy-ayl//2):cy+ayl-ayl//2, max(0,cz-azl//2):cz+azl-azl//2] = output_poly_array[max(0,axl//2-cx):axl//2+mask.shape[0]-cx,max(0,ayl//2-cy):ayl//2+mask.shape[1]-cy,max(0,azl//2-cz):azl//2+mask.shape[2]-cz]

    return mask

def pick_center(mask_scan):
    # we first find z index and then sample point with z slice
    labels = np.sort(np.unique(mask_scan))
    if labels[0] == 0:
        labels = np.delete(labels,0)
    
    # ana = random.choice(labels)
    # mask_scan = mask_scan==ana

    mask_scan = mask_scan>0

    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    z = round(random.uniform(0, 1) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    coordinates = np.argwhere(liver_mask)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)

    return xyz[0],xyz[1],xyz[2]

def find_position(edge_anatomy, center_anatomy, whole_brain, properties, abnormal_mask, abnormal_gen):

    mask = np.zeros(whole_brain.shape)
    abnormal_crop_gen = np.zeros(whole_brain.shape)

    nnunet_flag = False
    ### find where to put abnormal ###
    if 'class_locations' in properties.keys():

        # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
        foreground_classes = np.array(
            [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
        foreground_classes = foreground_classes[foreground_classes > 0]

        if len(foreground_classes) == 0:
            # this only happens if some image does not contain foreground voxels at all
            selected_class = None
            voxels_of_that_class = None
            print('case does not contain any foreground classes', i)
        else:
            selected_class = np.random.choice(foreground_classes)
            voxels_of_that_class = properties['class_locations'][selected_class]

        if voxels_of_that_class is not None:
            cx,cy,cz = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            nnunet_flag = True
        
    if not nnunet_flag:
        if np.random.rand() < 0.4:
            # is_edge = True
            cx,cy,cz = pick_center(edge_anatomy) 
        else:
            cx,cy,cz = pick_center(center_anatomy)
    
    brain_start_x,brain_end_x,brain_start_y,brain_end_y,brain_start_z,brain_end_z = crop(abnormal_mask)
    abnormal_mask = abnormal_mask[brain_start_x:brain_end_x,brain_start_y:brain_end_y,brain_start_z:brain_end_z]
    abnormal_gen = abnormal_gen[brain_start_x:brain_end_x,brain_start_y:brain_end_y,brain_start_z:brain_end_z]
    
    axl,ayl,azl = abnormal_mask.shape[0], abnormal_mask.shape[1], abnormal_mask.shape[2]
    
    mask[max(0,cx-axl//2):cx+axl-axl//2, max(0,cy-ayl//2):cy+ayl-ayl//2, max(0,cz-azl//2):cz+azl-azl//2] = abnormal_mask[max(0,axl//2-cx):axl//2+mask.shape[0]-cx,max(0,ayl//2-cy):ayl//2+mask.shape[1]-cy,max(0,azl//2-cz):azl//2+mask.shape[2]-cz]
    mask[whole_brain == 0] = 0

    abnormal_crop_gen[max(0,cx-axl//2):cx+axl-axl//2, max(0,cy-ayl//2):cy+ayl-ayl//2, max(0,cz-azl//2):cz+azl-azl//2] = abnormal_gen[max(0,axl//2-cx):axl//2+mask.shape[0]-cx,max(0,ayl//2-cy):ayl//2+mask.shape[1]-cy,max(0,azl//2-cz):azl//2+mask.shape[2]-cz]
    abnormal_crop_gen[whole_brain == 0] = 0

    return mask, abnormal_crop_gen, cx, cy, cz

def crop(data):
    mask = data.astype(np.int16)
    mask = (mask - mask.mean()) / (mask.std() + 1e-8)
    mask = mask >= 0
    zx = np.any(mask, axis=(1,2))
    start_slicex, end_slicex = np.where(zx)[0][[0, -1]]
    zy = np.any(mask, axis=(0,2))
    start_slicey, end_slicey = np.where(zy)[0][[0, -1]]   
    zz = np.any(mask, axis=(0,1))
    start_slicez, end_slicez = np.where(zz)[0][[0, -1]]
    return start_slicex, end_slicex, start_slicey, end_slicey, start_slicez, end_slicez

def get_texture(normal_img, img_array, tumor_mask):

    # if np.std(img_array) < 0.01: 
    #     print('nonzero std is 0!')
    #     img_array =  (img_array - np.min(img_array))/(np.max(img_array)-np.min(img_array))
    # else: 
    #     img_array = (img_array - np.mean(img_array))/ np.std(img_array)

    tumor_gen = img_array.copy()[np.newaxis,:,:,:]
    tumor_aug = np.array(trans(tumor_gen))[0]

    # tumor_aug = img_array

    tumor_aug[tumor_mask==0]=0

    # # tumor_mean = np.mean(tumor_aug[np.nonzero(tumor_aug)])
    if np.random.random()< 0.5:
        intensity_index = random.uniform(1.0,3.0)
        # tumor_aug = tumor_aug*(np.mean(normal_img[np.nonzero(normal_img)])*intensity_index/np.mean(img_array[np.nonzero(img_array)]))
        tumor_aug = tumor_aug * intensity_index
    return tumor_aug

def mix_array(normal,tumor,mask):
    # mix_ratio_value = random.uniform(0.2,0.8)
    # mix_ratio_value = 1
    # mix_mask = mask*mix_ratio_value
    # mix_array = np.multiply(mix_mask, tumor) + np.multiply((1-mix_mask), normal)

    # return mix_array, mask

    # temp = mask.astype(np.uint8)
    sigma = np.random.uniform(1,2)
    mix_mask = gaussian_filter(mask*1.0, sigma)

    values = mix_mask[mix_mask > 0]
    max_v = np.percentile(values,99)
    if max_v < 0.6:
        r = 0.6/max_v
        mix_mask = mix_mask * r
        # print("95 v",max_v,"r",r,"max",np.max(geo_blur),"min",np.min(geo_blur))
    temp = mix_mask >= 0.25
    
    mix_array = normal * (1 - mix_mask) + tumor * mix_mask

    #tumor[mask!=0] = mix_array[mask!=0]
    #return mix_array, tumor
    return mix_array, temp

def GetSingleLesion(brain_scan, edge_anatomy, center_anatomy, whole_brain, modality, properties, refpaths, name):

    # 1）初始化异常区域形状（用椭圆或者解剖区域形状均可），进行适当的放大缩小等变形；GetSingleLesion
    # abnormal_mask.shape is a whatevershape
    ref_path = refpaths[random.choice(list(refpaths.keys()))][modality]
    ref_img = sitk.ReadImage(ref_path)
    ref_img.SetOrigin((0.0, 0.0, 0.0))
    ref_array = sitk.GetArrayFromImage(ref_img)

    # get the abnormal mask shape = ref_array.shape
    abnormal_mask = tumor_array_generate(ref_array,ref_img,input_poly_data, name)

    # tumor_gen tumor_mask with a tumor texture shapfind_positione = ref_array.shape
    abnormal_gen = get_texture(brain_scan, ref_array, abnormal_mask)

    # 2）确定异常安放位置（非脑外，可以选择某个确定的解剖区域，也可以随机在脑内选择一个点）；
    # 一定几率选在脑部边缘，模拟硬膜血肿或者脑膜瘤
    # 一定几率在其他部位
    # abnormal_mask.shape normal_img.shape
    # abnormal_gen.shape normal_img.shape

    abnormal_mask, abnormal_gen, cx,cy,cz = find_position(edge_anatomy, center_anatomy, whole_brain, properties, abnormal_mask, abnormal_gen)
    
    brain_scan, abnormal_mask = mix_array(brain_scan,abnormal_gen,abnormal_mask)

    abnormal_mask[whole_brain == 0] = 0
    
    return brain_scan, abnormal_mask, [cx,cy,cz]

def seperate(anatomy_scan):

    z = anatomy_scan.shape[-1]

    whole_brain = anatomy_scan.copy()
    whole_brain[whole_brain>0] = 1
    for i in range(z):
        whole_brain[:,:,i] = binary_fill_holes(whole_brain[:,:,i]).astype(int)

    edge_width = 12
    kernel = np.ones((edge_width,edge_width), dtype=np.uint8)
    mask_dilate = cv2.dilate(whole_brain, kernel, iterations=1)
    whole_brain[np.logical_xor(mask_dilate,whole_brain)] = 2

    skull_gap = whole_brain == 2

    anatomy_scan_temp = whole_brain == 1

    edge_width = 15
    kernel = np.ones((edge_width,edge_width), dtype=np.uint8)
    anatomy_scan_temp.dtype = np.uint8
    erode_anatomy = cv2.erode(anatomy_scan_temp, kernel, iterations=1)
    anatomy_edge = np.logical_xor(anatomy_scan_temp, erode_anatomy)

    full_anatomy_scan =anatomy_scan.copy()
    full_anatomy_scan[np.logical_and(anatomy_scan==0, whole_brain==1)] = 100
    
    edge = full_anatomy_scan.copy()
    edge[np.logical_not(anatomy_edge)] = 0

    center = full_anatomy_scan.copy()
    center[np.logical_not(erode_anatomy)] = 0

    return edge, center, skull_gap, whole_brain==1

def SynthesisTumor(brain_scan, anatomy_scan, modality, properties, ref_paths, name):
    
    abnormal_mask = np.zeros(anatomy_scan.shape)
    num_lesions = random.randint(1,4)
    # print("num_lesions",num_lesions)
    # edge_anatomy, center_anatomy = seperate(anatomy_scan)
    edge_anatomy, center_anatomy, skull_gap, whole_brain = seperate(anatomy_scan)

    xyzs = []

    num_lesions = 1

    for cnt in range(num_lesions):
        brain_scan, abnormal_mask_anatomy, center_coords = GetSingleLesion(brain_scan, edge_anatomy, center_anatomy, whole_brain, modality, properties, ref_paths, name)
        abnormal_mask = np.logical_or(abnormal_mask, abnormal_mask_anatomy)
        xyzs.append(center_coords)

    if np.sum(abnormal_mask>0)<30:
        print("sum",np.sum(abnormal_mask))
        raise Exception

    return brain_scan,abnormal_mask, xyzs

if __name__=='__main__':
    pass