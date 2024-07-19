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
import importlib
import pkgutil

from .nnUNetTrainerV2_six_pub_seg import nnUNetTrainerV2 as nnUNetTrainerV2_six_pub_seg
from .nnUNetTrainerV2_llm_resize_new import nnUNetTrainerV2 as nnUNetTrainerV2_llm_resize


def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']

    print("init AutoRG_Brain_SEG model")
    trainer = nnUNetTrainerV2_six_pub_seg(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    checkpoint = join(folder, "model_best.model")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)


def load_model_and_checkpoint_files(folder, mixed_precision=None, checkpoint_name="model_best"):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """

    trainer = restore_model(join(folder, "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = [join(folder, "%s.model" % checkpoint_name)]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params

def load_model_and_checkpoint_files_llm(folder, mixed_precision=None, checkpoint_name="model_best"):
    
    trainer = restore_model_llm(join(folder, "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)

    all_best_model_files = [join(folder, "%s.model" % checkpoint_name)]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params

def restore_model_llm(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']

    print("init AutoRG_Brain_RG model")

    trainer = nnUNetTrainerV2_llm_resize(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        ## self.llm_model has loaded the params
        trainer.load_checkpoint(checkpoint, train)
    return trainer


if __name__ == "__main__":
    pass
