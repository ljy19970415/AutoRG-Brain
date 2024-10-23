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
from batchgenerators.utilities.file_and_folder_operations import *

def load_pretrained_weights_allow_missing(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        print("pretrain key",k)
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()

    # missing_keys = []
    # shape_not_ok = []
    # shape_ok = []
    # total = []
    # for key, _ in model_dict.items():
    #     print('my key',key)
    #     total.append(key)
    #     if key not in new_state_dict:
    #         print("missing",key)
    #     elif ('conv_blocks' in key):
    #         if model_dict[key].shape == pretrained_dict[key].shape:
    #             shape_ok.append(key)
    #             print("ok my shape",model_dict[key].shape,"pretrain shape",pretrained_dict[key].shape)
    #         else:
    #             shape_not_ok.append(key)
    #             print("not ok my shape",model_dict[key].shape,"pretrain shape",pretrained_dict[key].shape)
    # print("total",len(total),"miss keys",len(missing_keys),"shape not ok",len(shape_not_ok),"shape ok",len(shape_ok))

    # filter unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, _ in pretrained_dict.items():
            print(key)
    print("################### Done ###################")
    network.load_state_dict(model_dict)



def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        # print("pretrain key",k)
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict
    
    if torch.cuda.device_count() >1:
        model_dict = network.module.state_dict()
    else:
        model_dict = network.state_dict()
    
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                # print("my shape1",model_dict[key].shape,"pretrain shape",pretrained_dict[key].shape)
                continue
            else:
                # print("my shape2",model_dict[key].shape,"pretrain shape",pretrained_dict[key].shape)
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        if torch.cuda.device_count() >1:
            network.module.load_state_dict(model_dict)
        else:
            network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")

