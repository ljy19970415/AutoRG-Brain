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


from copy import deepcopy
from utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from .initialization import InitWeights_He
from .neural_network import SegmentationNetwork
import torch.nn.functional

from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
import math

import json

from batchgenerators.augmentations.utils import pad_nd_image

from typing import Union, Tuple, List
from torch.cuda.amp import autocast
from utilities.random_stuff import no_op
from utilities.to_torch import to_cuda, maybe_to_torch

from batchgenerators.utilities.file_and_folder_operations import *

import SimpleITK as sitk


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
        

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    # self.num_input_channels, self.base_num_features, 96, 2,
    # len(self.net_num_pool_op_kernel_sizes),
    # self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
    # dropout_op_kwargs,
    # net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
    # self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True

    def __init__(self, input_channels, base_num_features, num_classes_anatomy, num_classes_abnormal, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,feature_layer=0,size=256):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling # True
        self.convolutional_pooling = convolutional_pooling # True
        self.upscale_logits = upscale_logits # False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        # self.num_classes = num_classes
        self.num_classes_anatomy = num_classes_anatomy
        self.num_classes_abnormal = num_classes_abnormal
        
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context_a = []
        self.conv_blocks_context_b = []
        self.conv_blocks_context_c = []
        self.conv_blocks_context_d = []

        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs_anatomy = []
        self.seg_outputs_abnormal = []

        output_features = base_num_features
        input_features = input_channels

        # b = {}

        # num_pool = 6
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            # b[str(d)] = {"conv_args":self.conv_kwargs,"stride":first_stride}
            
            # add convolutions
            self.conv_blocks_context_a.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.conv_blocks_context_b.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.conv_blocks_context_c.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.conv_blocks_context_d.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            # if not self.convolutional_pooling:
            # not run here because self.convolutional_pooling = True
            self.td.append(pool_op(pool_op_kernel_sizes[d]))
            
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            # choose this
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            # choose this
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context_a[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        
        self.conv_blocks_context_a.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))
        self.conv_blocks_context_b.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))
        self.conv_blocks_context_c.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))
        self.conv_blocks_context_d.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))
        

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context_a[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context_a[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                # choose this
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))
        
        ##### the anatomy segmentation output ####
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs_anatomy.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes_anatomy,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        
        ##### the abnormal segmentation output ####
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs_abnormal.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes_abnormal,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                # choose this
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization) # decoder
        self.conv_blocks_context_a = nn.ModuleList(self.conv_blocks_context_a) # encoder_a
        self.conv_blocks_context_b = nn.ModuleList(self.conv_blocks_context_b) # encoder_b
        self.conv_blocks_context_c = nn.ModuleList(self.conv_blocks_context_c) # encoder_c
        self.conv_blocks_context_d = nn.ModuleList(self.conv_blocks_context_d) # encoder_d

        self.td = nn.ModuleList(self.td) # MaxPool3D s
        self.tu = nn.ModuleList(self.tu) # nn.ConvTranspose3d s
        self.seg_outputs_anatomy = nn.ModuleList(self.seg_outputs_anatomy)
        self.seg_outputs_abnormal = nn.ModuleList(self.seg_outputs_abnormal)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)
        
        ###### get area feature from decoder layer #####
        
        # feature_layer

        self.feature_layer = feature_layer

        ### Task002_six
        # self.feature_dim_array = [320,320,256,128,64,32]
        # self.area_dim_array = [[5,8,8],[10,16,16],[20,32,32],[20,64,64],[20,128,128],[20,256,256]]

        ### Task008_seg
        self.feature_dim_array = [320,256,128,64,32]
        self.area_dim_array = [[7,8,7],[14,16,14],[28,32,28],[56,64,56],[112,128,112]]


        ##### two conv layers fold2 ###
        # if self.size == 256:
        z,x,y = self.area_dim_array[self.feature_layer]
        # else:
        #     z,x,y = 20,64,64
        input_features = self.feature_dim_array[self.feature_layer]
        output_features = input_features
        self.pool_conv = []

        ## Task002 six
        # for u in range(100):
        #     z,x,y = math.ceil(z/2),math.ceil(x/2),math.ceil(y/2)
            
        #     if u!= 0:
        #         output_features *= 2
        #     self.pool_conv.append(
        #         StackedConvLayers(input_features, output_features, 1, self.conv_op, self.conv_kwargs,
        #                       self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
        #                       self.nonlin_kwargs, [2,2,2], basic_block=basic_block)
        #     )
        #     input_features = output_features
        #     if output_features == 1024:
        #         break

        ### Task 008 seg
        if size == 4:
            for u in range(100):

                z,x,y = math.ceil(z/2),math.ceil(x/2),math.ceil(y/2)
                
                output_features *= 2
                self.pool_conv.append(
                    StackedConvLayers(input_features, output_features, 1, self.conv_op, self.conv_kwargs,
                                self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                self.nonlin_kwargs, [2,2,2], basic_block=basic_block)
                )
                
                input_features = output_features
                if output_features == 1024:
                    break
        elif size == 8:
            self.conv_kwargs['padding'] = [0,1,0]
            for u in range(10):
                output_features *= 2
                
                if u ==0:
                    factor = 1
                    self.conv_kwargs["kernel_size"] = [2,3,2]
                else:
                    factor = 2
                
                # factor = 2

                z = (z-self.conv_kwargs["kernel_size"][0]+2*self.conv_kwargs['padding'][0])//factor+1
                x = (x-self.conv_kwargs["kernel_size"][1]+2*self.conv_kwargs['padding'][1])//factor+1
                y = (y-self.conv_kwargs["kernel_size"][2]+2*self.conv_kwargs['padding'][2])//factor+1
                
                self.pool_conv.append(
                    StackedConvLayers(input_features, output_features, 1, self.conv_op, self.conv_kwargs,
                                self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                self.nonlin_kwargs, [factor,factor,factor], basic_block=basic_block)
                )
                input_features = output_features
                if output_features == 1024:
                    break
        
        self.pool_conv = nn.ModuleList(self.pool_conv)
        
        self.img_patch_num = z*x*y
        
        # conv_kwargs_first_conv = deepcopy(self.conv_kwargs)
        # conv_kwargs_first_conv['stride'] = self.pool_stride

        ### one layer conv fold 1 ###
        # self.pool_conv = ConvDropoutNormNonlin(input_features, input_features, self.conv_op,
        #                    conv_kwargs_first_conv,
        #                    self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
        #                    self.nonlin, self.nonlin_kwargs)

        # self.area_feature_dim = self.feature_dim_array[self.feature_layer]
    
    def cal_area_feature_dim(self):

        d = self.feature_dim_array[self.feature_layer]

        z,x,y = self.area_dim_array[self.feature_layer] if self.pool_to_feature_layer is None else self.area_dim_array[self.pool_to_feature_layer]

        if self.avg_type == 'z':
            return d*x*y
        elif self.avg_type == 'xyz':
            return d
        elif self.avg_type == 'no':
            return d*x*y*z
    
    def forward(self, x, target, modal, region, eval_mode_for_six='global',choose_dataset=None):

        # region shape [b,num_regions_in_each_image] [[[21],[23,24],[9]],[[5],[9]]]
        # target b, 2, patch_size, with target[:,0,:] is anatomy target target[:,1,:] is abnormal target

        # a = {}
        # a['x1']=list(x.shape)
        # a['target1'] = list(target.shape)

        if not x.shape[2] < x.shape[3] and not x.shape[2] < x.shape[4]:
            x = x.permute(0,1,4,2,3)
            target = target.permute(0,1,4,2,3)

        # a['x2']=list(x.shape)
        # a['target2'] = list(target.shape)
        
        only_one_target = True if target.shape[1] == 1 else False
        
        # target_anatomy = target[:,:-1,:] if not only_one_target else target[:,-1:,:]
        # target_abnormal = target[:,-1:,:]
        
        skips = []

        if modal == 'DWI':
            conv_blocks_context = self.conv_blocks_context_a
        elif modal == 'T1WI':
            conv_blocks_context = self.conv_blocks_context_b
        elif modal == 'T2WI':
            conv_blocks_context = self.conv_blocks_context_c
        elif modal == 'T2FLAIR':
            conv_blocks_context = self.conv_blocks_context_d
        
        # a["pool_op_kernel_sizes"] = self.pool_op_kernel_sizes
        # a["-1"]=list(x.shape)
        
        for d in range(len(conv_blocks_context) - 1):
            x = conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                # not run here because self.convolutional_pooling = False
                x = self.td[d](x)

        x = conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

            if u == self.feature_layer:
                break
        
        for u in range(len(self.td)-1-self.feature_layer):
            target = self.td[u](target)
            # a['ana_'+str(u)]=list(target_anatomy.shape)

        region_features = []

        # x_in b, 1, 224, 224, 32 
        # regions: ["x11,y11,z11,x21,y21,z21;x12,y12,z12,x22,y22,z22",""] total b elements
        # report: ["sdsfjlfj;sdjksjd",""] total b elements

        # target b, 2, z, x, y
        # data b, d, z, x, y
        
        # region shape [b,num_regions_in_each_image] [[[21],[23,24],[9]],[[5],[9]]]

        for b in range(len(x)):
            r = region[b]
            img_feature = x[b] # d, z, x, y

            b_target = target[b][:-1] if not only_one_target else target[b][-1:] # 1, z, x, y
            a_target = target[b][-1:]
            
            ana_masks = []
            for ana_group in r:
                if ana_group!="global":
                    ana_masks.append(np.zeros(b_target.shape))
                    for ana in ana_group:
                        ana_masks[-1] = np.logical_or(ana_masks[-1],b_target==ana)
                else:
                    ana_masks.append(a_target)

            ### get the individual bbox abnormal areas ###
            
            bboxes_ab = sk_regions(sk_label(a_target[0]))

            for idx,ana_group in enumerate(r):

                if ana_group == "global":

                    abnormal = np.ones(b_target.shape)

                else:

                    abnormal = np.zeros(b_target.shape)

                    for box in bboxes_ab:

                        z1,x1,y1,z2,x2,y2 = box.bbox
                        
                        if len(np.where(ana_masks[idx][:,z1:z2+1,x1:x2+1,y1:y2+1])[0]):
                            abnormal[:,z1:z2+1,x1:x2+1,y1:y2+1] = 1
                    
                    abnormal = np.logical_or(abnormal,ana_masks[idx])

                abnormal = torch.tensor(abnormal, dtype=torch.float16)

                abnormal = abnormal.repeat(img_feature.shape[0],1,1,1).to(img_feature.device) # d, z, x, y

                abnormal_feature = abnormal * img_feature # d, z, x, y

                cnt = 0
                for pool_layer in self.pool_conv:
                    abnormal_feature = pool_layer(abnormal_feature)
                    cnt+=1

                abnormal_feature = torch.flatten(abnormal_feature,start_dim=1,end_dim=3)

                abnormal_feature = abnormal_feature.permute(1,0)

                region_features.append(abnormal_feature)


        ## should use deep_supervision_in_scales in plans to maxpool target
        ## "deep_supervision_scales": "[[1, 1, 1], [1.0, 0.5, 0.5], [1.0, 0.25, 0.25], [1.0, 0.125, 0.125], [0.5, 0.0625, 0.0625], [0.25, 0.03125, 0.03125]]"
        ## reference augmentation/downsampling.py: def downsample_seg_for_ds_transform2

        return region_features,[]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
    
    ### below is newly added for test llm ###

    def predict_3D(self, x: np.ndarray, s, region, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True, modal=None, eval_mode="region_oracle"):
        
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')
 
        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    # if use_sliding_window:
                    #     res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                    #                                                  regions_class_order, use_gaussian, pad_border_mode,
                    #                                                  pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                    #                                                  verbose=verbose, modal=modal)
                    # else:
                    region_features,region_direction_names = self._internal_predict_3D_3Dconv(x, s, region, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose, modal=modal)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return region_features,region_direction_names

    def _internal_predict_3D_3Dconv(self, x: np.ndarray, s, region, min_size: Tuple[int, ...], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True, modal=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_3D_3Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        seg, _ = pad_nd_image(s, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        region_features,region_direction_names = self._internal_maybe_mirror_and_pred_3D(data[None], seg[None], region, mirror_axes, do_mirroring,
                                                                          modal=modal)


        return region_features,region_direction_names

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], s, region, mirror_axes: tuple,
                                           do_mirroring: bool = True, modal=None):
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)

        if s is not None:
            s = maybe_to_torch(s)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            # if s is not None:
            #     s = to_cuda(s, gpu_id=self.get_device())

        region_features,region_direction_names = self(x, s, modal, region)

        return region_features,region_direction_names
