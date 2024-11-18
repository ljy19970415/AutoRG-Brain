import yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import tqdm
import os
from einops import rearrange
from transformers import AutoModel
from batchgenerators.utilities.file_and_folder_operations import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,confusion_matrix,average_precision_score

from time import time
from typing import Tuple, Union, List
import sys

if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

from multiprocessing import Pool
from network_training.model_restore import load_model_and_checkpoint_files_llm
from network_training.model_restore import load_model_and_checkpoint_files
from utilities.llm_metric import *
from run.load_pretrained_weights import *
from dataset.utils import nnUNet_resize
from utilities.nd_softmax import *
import uuid
from inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti

import SimpleITK as sitk

from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions

import re

class AutoRG_Brain():
    def __init__(self, gpu_id, config):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # config = yaml.load(open(inference_cfg, 'r'), Loader=yaml.Loader)

        self.trainer, params = load_model_and_checkpoint_files_llm(config['llm_folder'], mixed_precision=True,
                                                      checkpoint_name=config['llm_chk'])
        
        self.segmodel = SegModel(config)

        self.trainer.load_checkpoint_ram(params[0], False)
        # load_pretrained_weights(self.trainer.network, config['seg_pretrained'])
        load_pretrained_weights(self.trainer.network, join(config['seg_folder'],config['seg_chk']+'.model'))
        self.num_threads_preprocessing = 6
        self.step_size = 0.5
        self.mixed_precision = True
        self.num_threads_nifti_save = 2

        self.hammer_anas = json.load(open('utils_file/hammer_anas.json','r'))
        self.eval_mode = config['eval_mode']
    
    def report(self, input_case_dict):

        test_file = input_case_dict
        
        list_of_lists = [[j['image']] for j in test_file]
        list_of_ab_segs = [j['label'] if 'label' in j else None for j in test_file]
        list_of_ana_segs = [j['label2'] if 'label2' in j else None for j in test_file]
        list_of_reports = None if 'report' not in test_file[0] else [j['report'] for j in test_file]
        modals = [j['modal'] for j in test_file]

        list_of_ab_segs, list_of_ana_segs = self.segmodel.seg(list_of_lists, list_of_ab_segs, list_of_ana_segs, modals)

        case_identifiers = []
        for idx,j in enumerate(test_file):
            case_identifiers.append(j['image'].split('/')[-1].split('.')[0])
        
        pool = Pool(self.num_threads_nifti_save)
        
        print("emptying cuda cache")
        torch.cuda.empty_cache()

        preprocessing = preprocess_multithreaded(self.trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, self.num_threads_preprocessing)

        pred_report = []
        
        setup_seed(42)

        self.trainer.network.eval()
        self.trainer.llm_model.eval()

        # the_idx = 0
        result_se_mask = [None for _ in modals]

        result_modal = []
        for preprocessed in preprocessing:

            # identifier, modal, (r, d, s_ab, s_ana, dct) = preprocessed
            identifier, modal, the_image_path, the_ab_seg_path, the_ana_seg_path, (r, d, s_ab, s_ana, dct) = preprocessed

            d = np.expand_dims(nnUNet_resize(d[0],self.trainer.patch_size,axis=0),axis=0)
            s_ab = nnUNet_resize(s_ab[0], self.trainer.patch_size,is_seg=True,axis=0) if s_ab is not None else np.zeros(self.trainer.patch_size)
            s_ab = np.expand_dims(s_ab, axis=0)
            s_ana = nnUNet_resize(s_ana[0], self.trainer.patch_size,is_seg=True,axis=0) if s_ana is not None else np.zeros(self.trainer.patch_size)
            s_ana = np.expand_dims(s_ana, axis=0)
            s = np.concatenate((s_ana,s_ab),axis=0)

            if r is not None:
                regions, gt_reports = self.trainer.split_batch_report([r])
            else:
                regions, gt_reports = None, None

            region_features, region_direction_names  = self.trainer.predict_preprocessed_data_return_region_report(
                d, s, regions, do_mirroring=False, mirror_axes=self.trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=self.step_size, use_gaussian=True, all_in_gpu=False,
                mixed_precision=self.mixed_precision, modal=modal, eval_mode=self.eval_mode)

            region_features = torch.tensor(np.array([item.cpu().detach().numpy() for item in region_features]), dtype=torch.float32).to(self.trainer.llm_model.device)
            
            output = self.trainer.llm_model.generate(
                    region_features,
                    max_length=300,
                    num_beams=1,
                    num_beam_groups=1,
                    do_sample=False,
                    num_return_sequences = 1,
                    early_stopping=True
            )
            del region_features

            generated_sents_for_selected_regions = self.trainer.tokenizer.batch_decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            ## the index of the sentence mask ##
            if self.eval_mode == "region_segtool":
                pred_global_report = generated_sents_for_selected_regions[-1]

                pred_region_concat_report = []
                for cur_idx, se in enumerate(generated_sents_for_selected_regions[:-1]):
                    # no anatomy is mentioned in the sentence
                    print("sentence",generated_sents_for_selected_regions[cur_idx],"anatomy",region_direction_names[cur_idx])
                    ana_flag = False
                    for cur_a in self.hammer_anas:
                        if cur_a in se.lower():
                            ana_flag=True
                            break
                    # anatomy is mentioned in the sentence
                    if not ana_flag:
                        sort_ana = sorted(region_direction_names[cur_idx][1],key =lambda x:-x[0])
                        most_pixel = sort_ana[0][0]
                        sort_ana = list(filter(lambda x:x[0]>=most_pixel, sort_ana))
                        ana_str = ' and '.join([item[1] for item in sort_ana])
                        se = se.strip()
                        se = se[:-1] + ' in '+ana_str+'.' if se[-1] == '.' or se[-1] == ',' else se + ' in '+ana_str+'.'
                    else:
                        if region_direction_names[cur_idx][0] == "left":
                            se = se.replace('right','left')
                        elif region_direction_names[cur_idx][0] == "right":
                            se = se.replace('left','right')
                    pred_region_concat_report.append(se)
                    
                pred_region_concat_report = " ".join(pred_region_concat_report)

                left_sentence = ""

                pred_split = pred_global_report.split('.')
                pred_split_2 = []
                for se in pred_split:
                    pred_split_2.extend(se.split(','))
                pred_split = list(map(lambda x:x+'.',pred_split_2))
                
                if 'ventricle' not in pred_region_concat_report.lower() and 'ventricle' not in left_sentence.lower():
                    left_sentence = left_sentence+" ".join([g for g in pred_split if 'ventricle' in g.lower()])            
                if 'midline' not in pred_region_concat_report.lower() and 'midline' not in left_sentence.lower():
                    left_sentence = left_sentence+" "+" ".join([g for g in pred_split if 'midline' in g.lower()])
                if 'sulci' not in pred_region_concat_report.lower() and 'midline' not in left_sentence.lower():
                    left_sentence = left_sentence+" "+" ".join([g for g in pred_split if 'sulci' in g.lower()])
                
                if 'midline' not in pred_region_concat_report.lower() and 'midline' not in left_sentence.lower():
                    left_sentence += " No midline shift."
                    
                pred_region_concat_report +=" "+left_sentence
                
                pred_report.append({'image':the_image_path,'pred_report':pred_region_concat_report,'ab_mask':the_ab_seg_path,'ana_mask':the_ana_seg_path})

            elif self.eval_mode == "given_mask":
                pred_region_concat_report = " ".join(generated_sents_for_selected_regions)
                pred_report.append({'image':the_image_path,'pred_report':pred_region_concat_report,'ab_mask':the_ab_seg_path,'ana_mask':the_ana_seg_path})

        print("inference done.")

        pool.close()
        pool.join()
        
        return pred_report

class SegModel():
    def __init__(self, config):
        self.trainer, params = load_model_and_checkpoint_files(config['seg_folder'], mixed_precision=True,
                                                        checkpoint_name=config['seg_chk'])
        self.trainer.load_checkpoint_ram(params[0], False)
        self.num_threads_nifti_save = 2
        self.num_threads_preprocessing = 6
        self.output_mask_dir = config['output_dir']

    def seg(self, list_of_lists, list_of_ab_segs, list_of_ana_segs, modals):
        pool = Pool(self.num_threads_nifti_save)
        results = []

        output_ab_filenames = []
        output_ana_filenames = []
        ab_flag = []
        ana_flag = []
        for idx,item in enumerate(list_of_lists):
            # uid = uuid.uuid1().hex
            img_path = item[0].split('/')[-1].split('.')[0]
            if list_of_ab_segs[idx] is not None:
                ab_flag.append(True)
                output_ab_filenames.append(list_of_ab_segs[idx])
            else:
                ab_flag.append(False)
                output_ab_filenames.append(join(self.output_mask_dir,img_path+'_ab.nii.gz'))
            if list_of_ana_segs[idx] is not None:
                ana_flag.append(True)
                output_ana_filenames.append(list_of_ana_segs[idx])
            else:
                ana_flag.append(False)
                output_ana_filenames.append(join(self.output_mask_dir,img_path+'_ana.nii.gz'))
        
        print("emptying cuda cache")
        torch.cuda.empty_cache()

        if 'segmentation_export_params' in self.trainer.plans.keys():
            force_separate_z = self.trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = self.trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = self.trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
        
        print("starting preprocessing generator")
        # under 3dfullres setting, seg_from_prev_stage is None
        preprocessing = preprocess_multithreaded_seg(self.trainer, list_of_lists, modals, output_ab_filenames, output_ana_filenames, ab_flag, ana_flag, self.num_threads_preprocessing)
        print("starting prediction...")

        result_modal = []

        for preprocessed in preprocessing:

            output_ab_filename, output_ana_filename, is_exist_ab, is_exist_ana, image_path, modal, (d, s, dct) = preprocessed

            if is_exist_ab and is_exist_ana:
                continue

            print("predicting", output_ab_filename, output_ana_filename, "modal",modal)
            
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            # load the params of the network

            softmaxs = self.trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=False, mirror_axes=self.trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=0.5, use_gaussian=True, all_in_gpu=False,
                mixed_precision=True, modal=modal)
            
            softmax_abnormal, softmax_anatomy = softmaxs[1], softmaxs[3]

            transpose_forward = self.trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = self.trainer.plans.get('transpose_backward')
                softmax_abnormal = softmax_abnormal.transpose([0] + [i + 1 for i in transpose_backward]) # 2, x ,y z
                softmax_anatomy = softmax_anatomy.transpose([0] + [i + 1 for i in transpose_backward]) # 96, x, y, z
            
            # if list_of_segs is not None:
            #     gt = s[0]
            #     gt[gt<0] = 0

            #     pred = softmax_abnormal.argmax(0)
            #     gt[gt>0] = 1
            #     gt[gt<0] = 0
            
            npz_file = None

            if hasattr(self.trainer, 'regions_class_order'):
                region_class_order = self.trainer.regions_class_order
            else:
                region_class_order = None

            if not is_exist_ab:
                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                ((softmax_abnormal, output_ab_filename, dct, interpolation_order, region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z),)
                                                    ))
            if not is_exist_ana:
                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                ((softmax_anatomy, output_ana_filename, dct, interpolation_order, region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z, True),)
                                                ))

        pool.close()
        pool.join()

        return output_ab_filenames, output_ana_filenames

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_save_to_queue_seg(preprocess_fn, q, list_of_lists, modals, output_ab_files, output_ana_files, ab_flags, ana_flags, transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_ab_file = output_ab_files[i]
            output_ana_file = output_ana_files[i]
            ab_flag = ab_flags[i]
            ana_flag = ana_flags[i]
            print("preprocessing", l)
            d, s, dct = preprocess_fn(l, None, target_shape=None)

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            modal = modals[i]
            q.put((output_ab_file, output_ana_file, ab_flag, ana_flag, l, modal, (d, s, dct)))
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


def preprocess_multithreaded_seg(trainer, list_of_lists, modals, output_ab_files, output_ana_files, ab_flags, ana_flags, num_processes=2):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    # classes = list(range(1, trainer.num_classes)) # 96

    # assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []

    for i in range(num_processes):

        pr = Process(target=preprocess_save_to_queue_seg, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            modals[i::num_processes],
                                                            output_ab_files[i::num_processes], output_ana_files[i::num_processes], ab_flags[i::num_processes], ana_flags[i::num_processes], trainer.plans['transpose_forward']))
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
                p.terminate()
            p.join()

        q.close()
    
def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, transpose_forward):

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            the_ab_seg = list_of_ab_segs[i] if list_of_ab_segs is not None else None
            the_ana_seg = list_of_ana_segs[i] if list_of_ana_segs is not None else None

            target_shape = None

            if the_ab_seg is not None:
                d, s_ab, dct = preprocess_fn(l, the_ab_seg, target_shape=target_shape)
            else:
                s_ab = None
            
            if the_ana_seg is not None:
                d, s_ana, dct = preprocess_fn(l, the_ana_seg, target_shape=target_shape)
            else:
                s_ana = None

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                # np.save(output_file[:-7] + ".npy", d)
                # d = output_file[:-7] + ".npy"
                print(l)
            r = list_of_reports[i] if list_of_reports is not None else None
            identi = case_identifiers[i]
            modal = modals[i]
            q.put((identi, modal, l, the_ab_seg, the_ana_seg, (r, d, s_ab, s_ana, dct)))
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


def preprocess_multithreaded(trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_processes=6):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    q = Queue(1)
    processes = []

    for i in range(num_processes):
        the_ab_segs = list_of_ab_segs[i::num_processes] if list_of_ab_segs is not None else None
        the_ana_segs = list_of_ana_segs[i::num_processes] if list_of_ana_segs is not None else None
        the_reports = list_of_reports[i::num_processes] if list_of_reports is not None else None
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes], the_ab_segs, the_ana_segs, the_reports,case_identifiers[i::num_processes],modals[i::num_processes],
                                                            trainer.plans['transpose_forward']))
        # pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
        #                                                     list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports,case_identifiers,modals,
        #                                                     trainer.plans['transpose_forward']))
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
                p.terminate()
            p.join()

        q.close()
