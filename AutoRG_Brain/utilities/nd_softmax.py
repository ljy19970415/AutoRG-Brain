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

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from skimage import metrics
import monai.metrics
from surface_distance import compute_surface_distances,compute_surface_dice_at_tolerance
from batchgenerators.utilities.file_and_folder_operations import *

def cal_dice(pred, true, spacing_mm=(1,1,1), tolerance=2):
    tp = np.sum(pred[true==1])
    fp = np.sum(pred[true==0])
    fn = np.sum(true[pred==0])
    tn = np.sum(np.logical_and(pred==0,true==0))
    dice = tp * 2.0 / (np.sum(pred) + np.sum(true))
    precision = tp * 1.0 / (tp+fp)
    sensitivity = tp * 1.0 / (tp+fn)
    specificity = tn * 1.0 / (tn+fp)

    surface_distances = compute_surface_distances(true.astype(bool), pred.astype(bool), spacing_mm=spacing_mm)
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return dice, nsd, precision, sensitivity, specificity

def simple_cal_dice(pred,true,mode='ab'):

    # tp = np.sum(pred[true==1])
    # fp = np.sum(pred[true==0])
    # fn = np.sum(true[pred==0])
    # dice = tp * 2.0 / (2*tp+fp+fn)   
    num_classes = 2 if mode == 'ab' else 96
    axes = tuple(range(0, len(true.shape))) # 1,2,3
    tp_hard = np.zeros(num_classes - 1)# b,c
    fp_hard = np.zeros(num_classes - 1)
    fn_hard = np.zeros(num_classes - 1)
    # if mode == 'ana':
    #     a['tp_hard1']=list(tp_hard.shape)
    for c in range(1, num_classes):
        tp_hard[c - 1] = np.sum((pred == c) * (true == c)) # b,c
        fp_hard[c - 1] = np.sum((pred == c) * (true != c))
        fn_hard[c - 1] = np.sum((pred != c) * (true == c))
    # if mode == 'ana':
    #     a['tp_hard2']=list(tp_hard.shape)
    dice = np.mean(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
    if mode == 'ana':
        a={}
        a['dice']=dice
        a['tp_hard']=list(tp_hard.shape)
        a['fp_hard']=list(fp_hard.shape)
        a['fn_hard']=list(fn_hard.shape)

    return dice

def cal_hd(pred, true):
    return metrics.hausdorff_distance(pred, true)

def cal_nsd(pred, true):
    return monai.metrics.compute_surface_dice(pred, true, class_thresholds = [],include_background=False, distance_metric='euclidean')

def cal_precision_2s(pred, true):
    
    confusion_matrix = monai.metrics.get_confusion_matrix(pred, true, include_background=True)
    precision = monai.metrics.compute_confusion_matrix_metric('precision', confusion_matrix)
    sensitivity = monai.metrics.compute_confusion_matrix_metric('sensitivity', confusion_matrix)
    specificity = monai.metrics.compute_confusion_matrix_metric('specificity', confusion_matrix)

    return precision, sensitivity, specificity

softmax_helper = lambda x: F.softmax(x, 1)

