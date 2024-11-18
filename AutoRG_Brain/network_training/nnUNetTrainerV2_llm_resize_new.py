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
from typing import Tuple
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
import torch.backends.cudnn as cudnn
from time import time
from tqdm import trange
import random
from .nnUNetTrainer_llm import nnUNetTrainer

# network
from network.generic_UNet_share_get_feature_patchwise_region import Generic_UNet as Generic_UNet_share_patch_region
from network.generic_UNet_share_get_feature_patchwise import Generic_UNet as Generic_UNet_share_patch

from network.initialization import InitWeights_He
from network.neural_network import SegmentationNetwork

# predict
import sys
if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

# augmentation

from augmentation.data_augmentation_moreDA_llm import get_moreDA_augmentation

from augmentation.data_augmentation_moreDA_llm_resize import get_moreDA_augmentation as get_moreDA_augmentation_noaug

from utilities.to_torch import maybe_to_torch, to_cuda
from utilities.tensor_utilities import sum_tensor
from augmentation.default_data_augmentation import get_patch_size, default_3D_augmentation_params

# loss
from loss.deep_supervision import MultipleOutputLoss2
from loss.dice_loss import DC_and_CE_loss

# lr
from lr.poly_lr import poly_lr
from torch.optim.lr_scheduler import ReduceLROnPlateau


# dataloader
from dataset.dataset_loading_llm import load_dataset, DataLoader3D, unpack_dataset
from dataset.dataset_loading_llm_bucket_resize_new import load_dataset_bucket, DataLoader3D_bucket, unpack_dataset_bucket

# utils
from utilities.nd_softmax import softmax_helper, cal_dice

import json
from torch.utils.tensorboard import SummaryWriter

## for test 
import SimpleITK as sitk

## llm
from transformers import GPT2Tokenizer
from network.language_model_patchwise import LanguageModel as LanguageModel_patch
from utilities.llm_metric import *

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, train_file, only_ana=False, abnormal_type="intense", num_batches_per_epoch=250, num_val_batches_per_epoch=50, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, network_type="normal",feature_layer=3,dataset_directory_bucket=None,
                 train_with_seg=False,avg_type='xyz',pool_to_feature_layer=None,use_conv_pool=False,use_global=True,size=4,dataset="six",max_tokens = 1024):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.init_args = (plans_file, fold, train_file, only_ana, abnormal_type, num_batches_per_epoch, num_val_batches_per_epoch, output_folder, dataset_directory, batch_dice, stage, 
        unpack_data, deterministic, fp16, network_type, feature_layer,dataset_directory_bucket,train_with_seg,avg_type,pool_to_feature_layer,use_conv_pool,use_global,size,dataset,max_tokens)
        
        self.epoch_rouges = []
        self.epoch_bleus = []
        self.epoch_f1s = []

        self.train_with_seg=train_with_seg

        self.plans_file = plans_file

        # self.train_file = None
        # self.test_file = None
        # self.report = {"training":[],"validation":[]}

        if not os.path.exists(train_file):
            train_file = None
        
        self.train_file = json.load(open(train_file,'r')) if train_file is not None else None
        self.test_file = self.train_file['validation'] if train_file is not None else None
        self.report = self.train_file['region_report'] if train_file is not None else None

        self.metric = ['rouge','bleu','f1']
        self.all_val_eval_metrics = {i:[] for i in self.metric}
        self.all_train_eval_metrics = {i:[] for i in self.all_val_eval_metrics}

        self.val_eval_criterion_MA = {i:None for i in self.all_val_eval_metrics}
        self.best_val_eval_criterion_MA = {i:None for i in self.all_val_eval_metrics}

        # self.only_ana = only_ana
        # self.abnormal_type = abnormal_type

        self.network_type = network_type

        self.feature_layer = feature_layer
        self.avg_type = avg_type
        self.pool_to_feature_layer = pool_to_feature_layer
        self.use_conv_pool = use_conv_pool
        self.use_global = use_global

        self.pin_memory = True
        
        try:
            tensorboard_output_dir = join(self.output_folder,'tensorboard')
            self.writer = SummaryWriter(tensorboard_output_dir)
            if not os.path.exists(tensorboard_output_dir):
                os.makedirs(tensorboard_output_dir)
        except:
            print("tensorboard writer initialize fails")
            self.writer = None
        
        self.num_batches_per_epoch = num_batches_per_epoch # 250
        self.num_val_batches_per_epoch = num_val_batches_per_epoch # 50
        self.save_every = 50

        self.unpack_data = True ## unpack the preprocessed .npz file to .npy file
        self.dataset_directory_bucket = dataset_directory_bucket

        self.client = None

        self.batch_size = 4
        # self.batch_size = 1 # use it when you use small dataset to debug

        self.size = size

        self.mode=dataset

        self.train_eval_epoch = 500
        self.val_eval_epoch = 5

        self.max_tokens = max_tokens

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
    
    def load_dataset_bucket(self):
        # if self.mode == "FT":
        #     dataset1 = load_dataset_bucket(self.folder_with_preprocessed_data,self.folder_with_preprocessed_data_bucket)
        #     dataset2 = load_dataset_bucket(self.folder_with_preprocessed_data_FT,self.folder_with_preprocessed_data_bucket_FT)
        #     self.dataset = {**dataset1, **dataset2}
        # elif self.mode == "public":
        #     self.dataset = load_dataset_bucket(self.folder_with_preprocessed_data_FT,self.folder_with_preprocessed_data_bucket_FT)
        # else:
        #     self.dataset = load_dataset_bucket(self.folder_with_preprocessed_data,self.folder_with_preprocessed_data_bucket)
        self.dataset = load_dataset_bucket(self.folder_with_preprocessed_data,self.folder_with_preprocessed_data_bucket)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, report=self.report["training"],
                                has_prev_stage=True, oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, report=self.report["validation"],
                                has_prev_stage=True,oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def get_basic_generators_bucket(self):
        self.load_dataset_bucket()
        self.do_split()
        dl_tr = DataLoader3D_bucket(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, report=self.report["training"],
                                has_prev_stage=True, oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r',client=self.client,dataset=self.mode)
        dl_val = DataLoader3D_bucket(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, report=self.report["validation"],
                                has_prev_stage=True,oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r',client=self.client,dataset=self.mode)
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False, no_aug=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.plans = load_pickle('utils_file/nnUNetPlansv2.1_plans_3D.pkl')
            self.process_plans(self.plans)

            ##### use the segmentation patch_size ####
            plans = load_pickle('utils_file/nnUNetPlansv2.1_plans_3D.pkl')
            self.patch_size = np.array(plans['plans_per_stage'][0]['patch_size']) # .astype(int) # 20 256 256

            ##### end use #########

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}) # batch_dice True
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            # self.folder_with_preprocessed_data_FT = "/mnt/petrelfs/leijiayu/nnUNet/nnUNet_preprocessed/Task010_llm_FT/nnUNetData_plans_v2.1_stage0"
            if self.dataset_directory_bucket is not None:
                self.folder_with_preprocessed_data_bucket = self.dataset_directory_bucket+'//'+self.plans['data_identifier'] + "_stage%d" % self.stage
                # self.folder_with_preprocessed_data_bucket_FT = "s3://leijiayu_nnunet//nnUNet_preprocessed//Task010_llm_FT//nnUNetData_plans_v2.1_stage0"
            
            ############## prepare test imgs and segs #############

            if training:
                if self.dataset_directory_bucket is None:
                    self.dl_tr, self.dl_val = self.get_basic_generators()
                else:
                    self.dl_tr, self.dl_val = self.get_basic_generators_bucket()

                if self.unpack_data:
                    print("unpacking dataset")
                    if self.dataset_directory_bucket is None:
                        unpack_dataset(self.folder_with_preprocessed_data,train_file = self.train_file)
                    else:
                        # the data is on the bucket
                        unpack_dataset_bucket(self.folder_with_preprocessed_data, self.folder_with_preprocessed_data_bucket, train_file = self.train_file, client=self.client)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                if no_aug:
                    self.tr_gen, self.val_gen = get_moreDA_augmentation_noaug(
                        self.dl_tr, self.dl_val,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        # deep_supervision_scales=self.deep_supervision_scales,
                        deep_supervision_scales=None,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )
                else:
                    self.tr_gen, self.val_gen = get_moreDA_augmentation(
                        self.dl_tr, self.dl_val,
                        self.data_aug_params[
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params,
                        # deep_supervision_scales=self.deep_supervision_scales,
                        deep_supervision_scales=None,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False
                    )
                
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass
            
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
    
    def get_tokenizer(self):
        checkpoint = "healx/gpt-2-pubmed-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        # {'batch_size': 2, 'num_pool_per_axis': [4, 5, 4], 'patch_size': array([112, 128, 112]), 
        # 'median_patient_size_in_voxels': array([160, 198, 180]), 'current_spacing': array([0.93699998, 0.93699998, 0.93699998]), 
        # 'original_spacing': array([0.93699998, 0.93699998, 0.93699998]), 'do_dummy_2D_data_aug': False, 
        # 'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 1]], 
        # 'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}

        if self.network_type == "normal":

            self.network = Generic_UNet(self.num_input_channels, self.base_num_features, 96, 2,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        else:
            ### use the segmentation model from fold 50, so here we load the plans from fold 50 and use it to initialize the network ###
            plans = load_pickle('utils_file/nnUNetPlansv2.1_plans_3D.pkl')
            num_input_channels = plans['num_modalities']
            base_num_features = plans['base_num_features']
            net_num_pool_op_kernel_sizes = plans['plans_per_stage'][0]['pool_op_kernel_sizes']
            conv_per_stage = plans['conv_per_stage']
            net_conv_kernel_sizes = plans['plans_per_stage'][0]['conv_kernel_sizes']
            
            if self.use_global:
                self.network = Generic_UNet_share_patch(num_input_channels, base_num_features, 96, 2,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,feature_layer=self.feature_layer, size=self.size)
            else:
                self.network = Generic_UNet_share_patch_region(num_input_channels, base_num_features, 96, 2,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,feature_layer=self.feature_layer, size=self.size)
        
        ### freeze the network parameter ####
        if not self.train_with_seg:
            for name, parameter in self.network.named_parameters():
                if name.startswith("pool_conv"):
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        
        # if torch.cuda.device_count() >1:
        #     self.llm_model = nn.DataParallel(LanguageModel_patch(self.network.img_patch_num))
        #     self.network = nn.DataParallel(self.network)
        # else:
        #     self.llm_model = LanguageModel_patch(self.network.img_patch_num, self.max_tokens)

        self.llm_model = LanguageModel_patch(self.network.img_patch_num, self.max_tokens)
        
        self.tokenizer = self.get_tokenizer()

        if torch.cuda.is_available():
            self.network.cuda()
            self.llm_model.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        # assert self.network is not None, "self.initialize_network must be called first"
        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)
        # #### this optimizer params seems better ####
        # self.optimizer.param_groups[0]["momentum"] = 0.95
        
        # self.lr_scheduler = None

        assert self.llm_model is not None, "self.llm_model must be called first"

        # if self.train_with_seg:
        #     self.optimizer = torch.optim.AdamW([{self.llm_model.parameters()}], lr=5e-5)
        # else:
        #     self.optimizer = torch.optim.AdamW(self.llm_model.parameters(), lr=5e-5)

        parameters = [{'params':self.llm_model.parameters()},{'params':self.network.parameters()}]
        self.optimizer = torch.optim.AdamW(parameters, lr=5e-5)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10, threshold=1e-3, cooldown=5)
        # mode = 'min' check if metric no longer decrease
        # mode = 'max' check if metric no longer increase
        # use: lr_scheduler.step(metric_value) to check input metric
        # factor when the triggered lr*=factor
        # patience: the wait amount of metric no changes to trigger lr change
        # threshold: only the change value >= threshold will be recognized as a change
        # cooldown: the wait epoch after a lr change, (avoid lr decreases quickly)
        # min_lr: set the minimum lr, lr won't below min_lr
        # eps: if the lr change gap less than 1e-8, then ignore this lr change


    def tokenized(self, report):
        # pharases shape (b,)
        # f = lambda x:x.split(';')
        # phrases = f(phrases) # List<List<str>>
        # temp = list(map(lambda x:x.split(';'), phrases))
        # phrases = np.asarray(phrases)
        # phrases = phrases.flatten()
        # phrases = []
        # for a in temp:
        #     phrases.extend(a)
        # phrases = list(np.asarray(phrases).flatten()) # List<str> shape [report_num_in_a_batch]

        phrases = []
        for a in report:
            phrases.extend(a)

        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        k = self.tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

        example = {}
        example["input_ids"] = k["input_ids"]
        example["attention_mask"] = k["attention_mask"]

        example = self.tokenizer.pad(example, padding="longest", return_tensors="pt")

        return example["input_ids"], example["attention_mask"], phrases

    def run_online_evaluation(self, output, reference_sents_for_selected_regions):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """

        with torch.no_grad():
            generated_sents_for_selected_regions = self.tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            rouges, bleus = compute_language_model_scores(generated_sents_for_selected_regions, reference_sents_for_selected_regions)
            bleu = np.mean(bleus)
            rouge = np.mean(rouges)
            f1 = 2*bleu*rouge/(bleu+rouge+1e-8)

            self.epoch_bleus.append(bleu)
            self.epoch_rouges.append(rouge)
            self.epoch_f1s.append(f1)

    def preprocess_save_to_queue(self, preprocess_fn, q, list_of_lists, list_of_segs, output_files, transpose_forward):
        # suppress output
        # sys.stdout = open(os.devnull, 'w')

        errors_in = []
        for i, l in enumerate(list_of_lists):
            try:
                output_file = output_files[i]
                print("preprocessing", output_file)
                d, s, dct = preprocess_fn(l,list_of_segs[i])
                # print(output_file, dct)
                
                """There is a problem with python process communication that prevents us from communicating objects 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                print(d.shape)
                if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                    print(
                        "This output is too large for python process-process communication. "
                        "Saving output temporarily to disk")
                    np.save(output_file[:-7] + ".npy", d)
                    d = output_file[:-7] + ".npy"
                q.put((output_file, (d, s, dct)))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print("error in", l)
                print(e)
        q.put("end")
        if len(errors_in) > 0:
            print("There were some errors in the following cases:", errors_in)
            print("These cases were ignored.")
        else:
            print("This worker has ended successfully, no errors to report")

    def preprocess_multithreaded(self, list_of_lists, list_of_segs, output_files, num_processes=6):

        # num_processes default = 6
        num_processes = min(len(list_of_lists), num_processes)

        # classes = list(range(1, trainer.num_classes)) # 96

        # assert isinstance(trainer, nnUNetTrainer)
        q = Queue(1)
        processes = []

        for i in range(num_processes):
            pr = Process(target=self.preprocess_save_to_queue, args=(self.preprocess_patient, q,
                                                                list_of_lists[i::num_processes],list_of_segs[i::num_processes],
                                                                output_files[i::num_processes], self.plans['transpose_forward']))
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()  # this should not happen but better safe than sorry right
                p.join()

            q.close()

    def validate(self, val_data, modal, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, use_gaussian: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        ####### insert the new code ########

        print("starting preprocessing generator")
        # under 3dfullres setting, seg_from_prev_stage is None
        preprocessing = self.preprocess_multithreaded(val_data['list_of_lists'], val_data['list_of_segs'], val_data['cleaned_output_files'])
        print("starting prediction...")

        dices = []

        for preprocessed in preprocessing:
            # s.shape 1,x,y,z
            # d.shape 1,x,y,z
            output_filename, (d, s, dct) = preprocessed

            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            print("predicting", output_filename)
            # load the params of the network
            # trainer.load_checkpoint_ram(params[0], False)

            # softmax.shape num_class, 144,174,138
            softmaxs = self.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_mirroring, mirror_axes=mirror_axes, use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=self.fp16, modal=modal) # get the [1], i.e., aggregated_results_abnormal
            
            softmax_abnormal, softmax_anatomy = softmaxs[1], softmaxs[3]

            # softmax_transpose.shape = num_classes, 144, 174, 138
            transpose_forward = self.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = self.plans.get('transpose_backward')
                softmax_abnormal = softmax_abnormal.transpose([0] + [i + 1 for i in transpose_backward])
                softmax_anatomy = softmax_anatomy.transpose([0] + [i + 1 for i in transpose_backward])
            
            gt = s[0]
            gt[gt<0] = 0

            if val_data['type'] == 'abnormal':
                # cal_abnormal dice
                gt[gt>0] = 1
                dices.append(cal_dice(softmax_abnormal.argmax(0), gt))
            else:
                # cal anatomy dice
                class_dice = []
                pred = softmax_anatomy.argmax(0)
                for ana in list(np.unique(gt)):
                    if ana == 0:
                        continue
                    class_dice.append(cal_dice(pred==ana, gt==ana))
                dices.append(np.mean(class_dice))

            # pred = softmax.argmax(0)
            # print("softmax transpose",pred.shape,list(np.unique(pred)),"d",d.shape,"gt",gt.shape,list(np.unique(gt)))

        
        ####### insert the new code #######

        # for k in self.dataset_val.keys():
        #     properties = load_pickle(self.dataset[k]['properties_file'])
        #     fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
        #     if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
        #             (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
        #         data = np.load(self.dataset[k]['data_file'])['data']

        #         print(k, data.shape)
        #         data[-1][data[-1] == -1] = 0

        #         softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
        #                                                                              do_mirroring=do_mirroring,
        #                                                                              mirror_axes=mirror_axes,
        #                                                                              use_sliding_window=use_sliding_window,
        #                                                                              step_size=step_size,
        #                                                                              use_gaussian=use_gaussian,
        #                                                                              all_in_gpu=all_in_gpu,
        #                                                                              mixed_precision=self.fp16)[1]

        #         softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        self.print_to_log_file("finished prediction")

        self.network.train(current_mode)
        self.network.do_ds = ds

        return np.mean(dices)

    def predict_preprocessed_data_return_region_report(self, data: np.ndarray, seg, region, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True, modal=None, eval_mode="region_oracle"):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']
        
        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        # self.llm_model.eval()
        ret = self.network.predict_3D(data, seg, region, modal=modal, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision, eval_mode=eval_mode)
        self.network.train(current_mode)
        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, modal, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True):
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']
        
        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, modal=modal, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret

    def do_split(self):
        
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """

        if self.train_file is not None:
            # total = json.load(open(self.train_file,'r'))
            tr_keys = self.train_file['training']
            val_keys = self.test_file
        elif self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]


        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        # self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
        #                                                      self.data_aug_params['rotation_y'],
        #                                                      self.data_aug_params['rotation_z'],
        #                                                      self.data_aug_params['scale_range'])
        self.basic_generator_patch_size = self.patch_size
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        # if epoch is None:
        #     ep = self.epoch + 1
        # else:
        #     ep = epoch
        # self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        # if self.lr_scheduler is not None:
            # assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            # if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            # lr scheduler is updated with moving average val loss. should be more robust
        
        self.lr_scheduler.step(self.train_loss_MA)
        # self.lr_scheduler.step(self.all_val_losses[-1])

        
        # self.lr_scheduler.step(self.epoch+1)
            #else:
            #self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

        # self.print_to_log_file("epoch",ep,"lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def finish_online_evaluation(self, mode="train"):

        if len(self.epoch_bleus):

            mean_bleu = np.mean(self.epoch_bleus)
            mean_rouge = np.mean(self.epoch_rouges)
            mean_f1 = np.mean(self.epoch_f1s)

            if mode == "train":
                self.all_train_eval_metrics["bleu"].append(mean_bleu)
                self.all_train_eval_metrics["rouge"].append(mean_rouge)
                self.all_train_eval_metrics["f1"].append(mean_f1)
            else:
                self.all_val_eval_metrics["bleu"].append(mean_bleu)
                self.all_val_eval_metrics["rouge"].append(mean_rouge)
                self.all_val_eval_metrics["f1"].append(mean_f1)
        
        for v in self.all_train_eval_metrics:
            if len(self.all_train_eval_metrics[v])==0:
                self.all_train_eval_metrics[v].append(0)
        for v in self.all_val_eval_metrics:
            if len(self.all_val_eval_metrics[v])==0:
                self.all_val_eval_metrics[v].append(0)

        
        self.epoch_bleus = []
        self.epoch_rouges = []
        self.epoch_f1s = []

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        # metrics

        # self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        self.manage_patience()

        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        # if self.epoch == 100:
        #     if self.all_val_eval_metrics_ana[-1] == 0 or self.all_val_eval_metrics_ab[-1] == 0:
        #         self.optimizer.param_groups[0]["momentum"] = 0.95
        #         self.network.apply(InitWeights_He(1e-2))
        #         self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
        #                                "high momentum. High momentum (0.99) is good for datasets where it works, but "
        #                                "sometimes causes issues such as this one. Momentum has now been reduced to "
        #                                "0.95 and network weights have been reinitialized")
        return continue_training
    
    def split_batch_report(self, region_report):
        region, report = [], []
        for d in region_report:
            report.append(list(d.keys()))
            region.append([d[i] for i in report[-1]])
        return region, report

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data'] # b,1,patch_size
        target = data_dict['target'] # we shut down deep_supervision; b, 2, patch_size, with target[:,0,:] is anatomy target target[:,1,:] is abnormal target

        modal = data_dict['modal']
        
        region, report = self.split_batch_report(data_dict['report']) # data_dict['report'] b, dct

        data = maybe_to_torch(data)
        # target_anatomy = maybe_to_torch(target_anatomy)
        # target_abnormal = maybe_to_torch(target_abnormal)

        if torch.cuda.is_available():
            data = to_cuda(data)
            # target_anatomy = to_cuda(target_anatomy)
            # target_abnormal = to_cuda(target_abnormal)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                # output_xxx is an array with length = unet depth, here len(target) = 5
                # output_xxx[0].shape b,num_classes,pathc_size
                # target[1].shape b,num_classes,patch_size//2 target[2].shape b,num_classes,patch_size//4 target[3].shape b,num_classes,patch_size//8 target[4].shape b,num_classes,patch_size//16
                a={}
                region_features, _ = self.network(data, target, modal, region, eval_mode="region_oracle")

                del data
                del region
                del target

                # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
                if torch.cuda.device_count() >1:
                    region_features = torch.tensor(np.array([item.cpu().detach().numpy() for item in region_features])).to(self.llm_model.module.device)
                else:
                    region_features = torch.tensor(np.array([item.cpu().detach().numpy() for item in region_features])).to(self.llm_model.device)

                #region_features = torch.tensor(np.array(region_features)).to(self.llm_model.device)

                ##### add end token begin token #####
                # bos_token (str, optional, defaults to "<|endoftext|>") — The beginning of sequence token.
                # eos_token (str, optional, defaults to "<|endoftext|>") — The end of sequence token. 
                ##### add end token begin token #####

                input_ids, attention_mask, reference_sents_for_selected_regions = self.tokenized(report) # input_ids/attention_mask shape [num_regions_in_batch x seq_len]
                
                oom = False 
                
                if torch.cuda.device_count() >1:
                    input_ids = input_ids.to(self.llm_model.module.device)
                    attention_mask = attention_mask.to(self.llm_model.module.device)
                else:
                    input_ids = input_ids.to(self.llm_model.device)
                    attention_mask = attention_mask.to(self.llm_model.device)

                try:
                    
                    language_model_loss = self.llm_model(input_ids, attention_mask, region_features, return_loss=True)

                    if run_online_evaluation:
                        output = self.llm_model.generate(
                                region_features,
                                max_length=300,
                                num_beams=1,
                                num_beam_groups=1,
                                do_sample=False,
                                num_return_sequences = 1,
                                early_stopping=True
                        )

                    del input_ids
                    del attention_mask
                    del region_features

                    if do_backprop:
                        self.amp_grad_scaler.scale(language_model_loss).backward()
                        self.amp_grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.llm_model.parameters(), 12)

                        ######## rethink the lr update here ############
                        self.amp_grad_scaler.step(self.optimizer)
                        self.amp_grad_scaler.update()

                except RuntimeError as e:  # out of memory error
                    if "out of memory" in str(e):
                        oom = True
                        raise e
                    else:
                        raise e
        else:
            pass
            # region_features = self.network(data, modal, region)
            # del data

            # if do_backprop:
            #     l.backward()
            #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            #     self.optimizer.step()
        
        if run_online_evaluation:
            self.run_online_evaluation(output, reference_sents_for_selected_regions)
            del output
        
        del reference_sents_for_selected_regions

        return language_model_loss.detach().cpu().numpy()

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        
        #self.maybe_update_lr()  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        
        
        # want at the start of the training
        # ds = self.network.do_ds
        # self.network.do_ds = True

        ## super run_training ##
        self.save_debug_information()
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)      
        # self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        
        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            
            # train one epoch
            self.network.train()
            self.llm_model.train()

            train_run_eval = True if (self.epoch+1) % self.train_eval_epoch == 0 else False
            val_run_eval = True if (self.epoch+1) % self.val_eval_epoch == 0 else False

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        # do_backdrop and run_online_evaluation

                        l = self.run_iteration(self.tr_gen, True, train_run_eval)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True, train_run_eval)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            ### this reset the online evaluation values so that it won't mess with 
            ### the later online evaluation on validation set (if it runs) 
            self.finish_online_evaluation(mode="train")

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                self.llm_model.eval()
                self.setup_seed(42)

                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, val_run_eval)
                    val_losses.append(l)
                
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])
                self.finish_online_evaluation(mode="val")
            
            # update value of self.train_loss_MA (in network_trainer.py)
            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

            ##### write tensorboard ######
            self.writer.add_scalars('loss',{'train':self.all_tr_losses[-1],"val":self.all_val_losses[-1]}, self.epoch)
            # record train dice

            for m in self.metric:
                self.writer.add_scalars('metric/'+m,{'train_'+m:self.all_train_eval_metrics[m][-1],'val_'+m:self.all_val_eval_metrics[m][-1]}, self.epoch)
            
            self.writer.add_scalar('lr',self.optimizer.param_groups[0]['lr'],self.epoch)

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
        ## run_training ##

        # self.network.do_ds = ds