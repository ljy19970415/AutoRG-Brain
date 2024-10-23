# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np

class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def augment_mirroring(self, sample_data, sample_seg=None, axes=(0, 1, 2)):
        if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
            raise Exception(
                "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
                "[channels, x, y] or [channels, x, y, z]")
        # if 0 in axes and np.random.uniform() < 0.5:
        #     sample_data[:, :] = sample_data[:, ::-1]
        #     if sample_seg is not None:
        #         sample_seg[:, :] = sample_seg[:, ::-1]

        # test
        # if 1 in axes and np.random.uniform() < 0.5:
        #     sample_data[:, :, :] = sample_data[:, :, ::-1]
        #     if sample_seg is not None:
        #         sample_seg[:, :, :] = sample_seg[:, :, ::-1]

        # if 2 in axes and len(sample_data.shape) == 4:
        #     if np.random.uniform() < 0.5:
        #         sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
        #         if sample_seg is not None:
        #             sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
        
        if 0 in axes and np.random.uniform() < 0.5:
            sample_data[:, :] = sample_data[:, ::-1]
            if sample_seg is not None:
                sample_seg[:, :] = sample_seg[:, ::-1]

        # 1 2
        if 1 in axes and np.random.uniform() < 0.5:
            sample_data[:, :, :] = sample_data[:, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :] = sample_seg[:, :, ::-1]

            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    
        return sample_data, sample_seg

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            ret_val = self.augment_mirroring(data[b], sample_seg, axes=self.axes)
            data[b] = ret_val[0]
            if seg is not None:
                seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict