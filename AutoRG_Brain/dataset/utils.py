import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
import os
import elasticdeform
from scipy.ndimage import gaussian_filter, binary_fill_holes
import cv2
import random
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from collections import OrderedDict
# from batchgenerator import *
from .batchgenerator import *
import json

def nnUNet_resize(data, new_shape, do_separate_z=True, is_seg=False, axis=2, order=3, order_z=0):
    assert len(data.shape) == 3, "data must be (x, y, z)"
    assert len(new_shape) == len(data.shape)

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
        order = 1
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            #print("separate z, order in z is", order_z, "order inplane is", order)
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []

            reshaped_data = []
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                elif axis == 1:
                    reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                else:
                    reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
            reshaped_data = np.stack(reshaped_data, axis)
            # print("reshaped_data",reshaped_data.shape)
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                reshaped_data = map_coordinates(reshaped_data, coord_map, order=order_z, mode='nearest').astype(dtype_data)
                
                #print("shape[axis] != new_shape[axis]",reshaped_data.shape)
                # else:
                #     unique_labels = np.unique(reshaped_data)
                #     reshaped = np.zeros(new_shape, dtype=dtype_data)

                #     for i, cl in enumerate(unique_labels):
                #         reshaped_multihot = np.round(
                #             map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                #                             mode='nearest'))
                #         reshaped[reshaped_multihot > 0.5] = cl
                #     reshaped_final_data.append(reshaped[None].astype(dtype_data))
            # else:
            #     reshaped_data = reshaped_data[None].astype(dtype_data)
        else:
            reshaped_data = resize_fn(data, new_shape, order, **kwargs).astype(dtype_data)
            # print("no separate z, order", reshaped_data.shape)
        return reshaped_data.astype(dtype_data)
    else:
        print("no resampling necessary",data.shape)
        return data

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def get_ellipsoid(x, y, z, n):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    n is the circle amount of the concentric circles

    """
    sh = (4*x, 4*y, 4*z)
    final_out = np.zeros(sh, int)

    old_x, old_y, old_z = x, y, z
    old_cx,old_cy,old_cz = 2*x,2*y,2*z
    for i in range(n):
        # calculate the ellipsoid 
        out = np.zeros(sh, int)
        aux = np.zeros(sh)
        if i == 0:
            radii = np.array([x, y, z])    
            com = np.array([2*x, 2*y, 2*z])  # center point
        else:
            new_x = random.randint(int(0.35*old_x), int(0.85*old_x))
            new_y = random.randint(int(0.35*old_y), int(0.85*old_y))
            new_z = random.randint(int(0.35*old_z), int(0.85*old_z))
            new_cx = random.randint(int(old_cx-old_x+new_x), int(old_cx+old_x-new_x))
            new_cy = random.randint(int(old_cy-old_y+new_y), int(old_cy+old_y-new_y))
            new_cz = random.randint(int(old_cz-old_z+new_z), int(old_cz+old_z-new_z))
            old_x, old_y, old_z = new_x, new_y, new_z
            old_cx, old_cy, old_cz = new_cx, new_cy, new_cz
            radii = np.array([new_x, new_y, new_z])
            com = np.array([new_cx,new_cy,new_cz])
        if not (radii > 0).all() or not (0<com[0]<4*x) or not (0<com[1]<4*y) or not (0<com[2]<4*z):
            break
        bboxl = np.floor(com-radii).clip(0,None).astype(int)
        bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
        roi = out[tuple(map(slice,bboxl,bboxh))]
        roiaux = aux[tuple(map(slice,bboxl,bboxh))]
        logrid = *map(np.square,np.ogrid[tuple(
                map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
        dst = (1-sum(logrid)).clip(0,None)
        mask = dst>roiaux

        roi[mask] = 1
        np.copyto(roiaux,dst,where=mask)
        final_out += out

    return final_out

def get_shape_from_anatomy(anatomy_scan):
    labels = np.sort(np.unique(anatomy_scan))
    if labels[0] == 0:
        labels = np.delete(labels,0)
    anatomy_shape = anatomy_scan==random.choice(labels)
    return anatomy_shape

def scale_shape(mask,alpha=(0., 1000.), sigma=(10., 13.),scale=(0.4,0.8),border_mode_seg='constant', border_cval_seg=0, order_seg=0):
    coords = create_zero_centered_coordinate_mesh(mask.shape)
    a = np.random.uniform(alpha[0], alpha[1])
    s = np.random.uniform(sigma[0], sigma[1])
    coords = elastic_deform_coordinates(coords, a, s)

    if np.random.random() < 0.5:
        sc = np.random.uniform(scale[0], 1)
    else:
        sc = np.random.uniform(1, scale[1])
    
    coords = scale_coords(coords, sc)

    # now find a nice center location 
    for d in range(3):
        ctr = mask.shape[d] / 2. - 0.5
        coords[d] += ctr

    mask = interpolate_img(mask, coords, order_seg, border_mode_seg, cval=border_cval_seg, is_seg=True)

    return mask

def get_shape(anatomy_scan):
    # mask = np.zeros((224,224,32))
    # cx, cy, cz = mask.shape[0]//2, mask.shape[1]//2, mask.shape[2]//2
    if np.random.rand() < 0.5:
        radius_dict = {0:[6,8],1:[8,15],2:[15,25],3:[25,40]}
        z_radius_dict = {0:[2,3],1:[3,5],2:[5,10],3:[10,15]}
        size = random.randint(0,3)
        x = random.randint(radius_dict[size][0],radius_dict[size][1])
        y = random.randint(radius_dict[size][0],radius_dict[size][1])
        z = random.randint(z_radius_dict[size][0],z_radius_dict[size][1])
        n = random.randint(1,4)
        abnormal_mask = get_ellipsoid(x, y, z, n)
    else:
        abnormal_mask = get_shape_from_anatomy(anatomy_scan)
        # if np.random.rand()<0.5:
        #     edge_width = random.randint(2,10)
        #     kernel = np.ones((edge_width,edge_width), dtype=np.uint8)
        #     abnormal_mask.dtype = np.uint8
        #     mask_dilate = cv2.dilate(abnormal_mask, kernel, iterations=1)
        #     abnormal_mask[np.logical_xor(mask_dilate,abnormal_mask)] = 2
        abnormal_mask = scale_shape(abnormal_mask)

    x_start, x_end = np.where(np.any(abnormal_mask!=0, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(abnormal_mask!=0, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(abnormal_mask!=0, axis=(0, 1)))[0][[0, -1]]
    abnormal_mask = abnormal_mask[x_start:x_end,y_start:y_end,z_start:z_end]

    return abnormal_mask

def get_intensity(brain_scan, anatomy_scan, modality):
    lateral_ventricle = np.logical_or(anatomy_scan == 46, anatomy_scan==47)
    lateral_ventricle_intensity = np.mean(brain_scan[lateral_ventricle])
    if modality == "T2WI":
        not_high_mask = brain_scan<lateral_ventricle_intensity
        brain_mask = np.logical_and(not_high_mask,np.logical_and(anatomy_scan!=0, np.logical_not(lateral_ventricle)))
        isointensity = np.percentile(brain_scan[brain_mask],30)
        thresh = 0.5
    else:
        not_low_mask = brain_scan>lateral_ventricle_intensity
        brain_mask = np.logical_and(not_low_mask,np.logical_and(anatomy_scan!=0, np.logical_not(lateral_ventricle)))
        isointensity = np.percentile(brain_scan[brain_mask],75)
        thresh = 0.5
    brain_max_value = np.max(brain_scan[anatomy_scan>0])
    brain_min_value = np.min(brain_scan[anatomy_scan>0])
    whole_max_value = np.max(brain_scan)
    low_gap = isointensity - brain_min_value
    high_gap = whole_max_value - isointensity
    # 0：低信号，1：等低信号，2：等信号，3：等高信号，4：高信号
    intensity_dic={ 0:[brain_min_value,brain_min_value+0.45*low_gap],\
                    1:[brain_min_value+0.45*low_gap,brain_min_value+0.9*low_gap],\
                    # 2:[isointensity-low_gap*0.1,isointensity+low_gap*0.1],\
                    2:[isointensity+low_gap*0.1,min(brain_max_value,isointensity+low_gap*0.1+0.75*high_gap)],\
                    3:[min(brain_max_value,isointensity+low_gap*0.1+0.75*high_gap),whole_max_value]}
    return intensity_dic, low_gap*thresh

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

def find_position(edge_anatomy, center_anatomy, whole_brain, abnormal_mask, properties):

    mask = np.zeros(whole_brain.shape)

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
    
    axl,ayl,azl = abnormal_mask.shape[0], abnormal_mask.shape[1], abnormal_mask.shape[2]
    mask[max(0,cx-axl//2):cx+axl-axl//2, max(0,cy-ayl//2):cy+ayl-ayl//2, max(0,cz-azl//2):cz+azl-azl//2] = abnormal_mask[max(0,axl//2-cx):axl//2+mask.shape[0]-cx,max(0,ayl//2-cy):ayl//2+mask.shape[1]-cy,max(0,azl//2-cz):azl//2+mask.shape[2]-cz]
    # if is_edge:
    #     mask[edge_anatomy == 0] = 0
    # else:
    #     mask[whole_brain == 0] = 0
    mask[whole_brain == 0] = 0

    return mask, cx, cy, cz

def get_texture(brain_scan, abnormal_mask, whole_brain, intensity_dic, gap, modality):

    abnormally_mask = np.zeros(abnormal_mask.shape)
    abnormally_full = brain_scan

    layer = np.sort(np.unique(abnormal_mask))
    # ignore background
    if layer[0] == 0:
        layer = np.delete(layer,0)

    # gen graduaaly high or low signals
    flag = 1
    textures = np.sort(np.random.randint(0,4,len(layer))) # gradually high
    # textures = list(map(lambda x:int(x), list(np.random.randint(0,4,len(layer)))))
    # textures.sort()
    if np.random.rand()<0.5:
        flag = 0 # gradually low
        textures = textures[::-1]
    
    for idx,i in enumerate(layer):

        temp = (abnormal_mask==i).astype(np.uint8)

        sigma = np.random.uniform(1, 2)
        geo_blur = gaussian_filter(temp*1.0, sigma)

        values = geo_blur[geo_blur > 0]
        max_v = np.percentile(values,99)
        if max_v < 0.6:
            r = 0.6/max_v
            geo_blur = geo_blur * r
        temp = geo_blur>=0.25

        mean_value = np.mean(brain_scan[abnormal_mask==i])
        legal_low = intensity_dic[textures[idx]][0]
        legal_high = intensity_dic[textures[idx]][1] 
        forbidden_low = mean_value - gap
        forbidden_high = mean_value + gap

        # print("texture",textures[idx])

        if legal_low <= forbidden_low and legal_high >= forbidden_high:
            new_idx = (textures[idx]+1)%4 if flag else (textures[idx]-1)%4
            legal_low = intensity_dic[new_idx][0]
            legal_high = intensity_dic[new_idx][1] 
        elif forbidden_high >= legal_low and forbidden_high <= legal_high:
            legal_low = forbidden_high
        elif forbidden_low >= legal_low and forbidden_low <= legal_high:
            legal_high = forbidden_low

        texture = legal_low + np.random.rand()*(legal_high-legal_low)

        # texture = gaussian_filter(texture * np.random.random_sample(temp.shape),0.8)

        abnormally = texture * geo_blur

        abnormally_full = abnormally_full * (1 - geo_blur) + abnormally
        abnormally_mask = np.logical_or(abnormally_mask, temp)
    
    # if is_edge:
    #     abnormally_mask[edge_anatomy==0] = 0
    # else:
    #     abnormally_mask[whole_brain==0] = 0
    abnormally_mask[whole_brain==0] = 0
    abnormally_full[abnormally_mask == 0] = brain_scan[abnormally_mask==0]

    return abnormally_full, abnormally_mask

def GetSingleLesion(brain_scan, anatomy_scan, edge_anatomy, center_anatomy, whole_brain, modality, properties):

    abnormal_mask = get_shape(anatomy_scan)

    abnormal_mask, cx,cy,cz = find_position(edge_anatomy, center_anatomy, whole_brain, abnormal_mask, properties)

    intensity_dic, gap = get_intensity(brain_scan, anatomy_scan, modality)

    brain_scan, abnormal_mask = get_texture(brain_scan, abnormal_mask, whole_brain, intensity_dic, gap, modality)
    
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

def SynthesisTumor(brain_scan, anatomy_scan, modality, properties=None):
    
    abnormal_mask = np.zeros(anatomy_scan.shape)
    num_lesions = random.randint(1,4)
    
    edge_anatomy, center_anatomy, skull_gap, whole_brain = seperate(anatomy_scan)

    xyzs = []

    for cnt in range(num_lesions):
        try:
            brain_scan, abnormal_mask_anatomy, center_coords = GetSingleLesion(brain_scan, anatomy_scan, edge_anatomy, center_anatomy, whole_brain, modality, properties)
            abnormal_mask = np.logical_or(abnormal_mask, abnormal_mask_anatomy)
            xyzs.append(center_coords)
        except:
            if cnt>0:
                print("continue")
                continue
            elif cnt == 0:
                raise Exception
    
    if np.sum(abnormal_mask>0)<30:
        print("sum",np.sum(abnormal_mask))
        raise Exception

    return brain_scan,abnormal_mask, xyzs


if __name__=='__main__':
    pass