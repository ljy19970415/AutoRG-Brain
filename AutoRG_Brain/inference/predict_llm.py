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

import random
import json
import numpy as np

from typing import Tuple, Union, List

import os

import sys
if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

import torch
from multiprocessing import Pool

from network_training.model_restore import load_model_and_checkpoint_files_llm
from utilities.llm_metric import *

from batchgenerators.utilities.file_and_folder_operations import *

from run.load_pretrained_weights import *

from dataset.utils import nnUNet_resize

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

            
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""

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
            q.put((identi, modal, (r, d, s_ab, s_ana, dct)))
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
    # restore output
    # sys.stdout = sys.__stdout__


def preprocess_multithreaded(trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_processes=2):

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

hammer_anas = json.load(open('utils_file/hammer_anas.json','r'))

def predict_cases(model, seg_pretrained, output_folder, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_threads_preprocessing,
                  num_threads_nifti_save, do_tta=True, 
                  mixed_precision=True,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  eval_mode="region_oracle"):
    """
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    
    pool = Pool(num_threads_nifti_save)
    
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    trainer, params = load_model_and_checkpoint_files_llm(model, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)

    trainer.load_checkpoint_ram(params[0], False)
    
    if seg_pretrained is not None:
        print("init AutoRG_Brain_SEG model")
        load_pretrained_weights(trainer.network, seg_pretrained)
    
    # if 'Radio_VQA' in list_of_lists[0][0] or 'radio' in list_of_lists[0][0]:
    #     trainer.plans['transpose_forward'] = [2,0,1]
    #     trainer.transpose_forward = [2,0,1]
    #     trainer.plans['transpose_backward'] = [1,2,0]
    #     trainer.transpose_backward = [1,2,0]

    print("starting preprocessing generator")
    # under 3dfullres setting, seg_from_prev_stage is None
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_threads_preprocessing)
    print("starting prediction...")

    if os.path.exists(os.path.join(output_folder,'pred_report.json')):
        pred_report = json.load(open(os.path.join(output_folder,'pred_report.json'),'r'))
    else:
        pred_report = {}

    trainer.network.eval()
    trainer.llm_model.eval()

    for preprocessed in preprocessing:

        identifier, modal, (r, d, s_ab, s_ana, dct) = preprocessed

        d = np.expand_dims(nnUNet_resize(d[0],trainer.patch_size,axis=0),axis=0)
        s_ab = nnUNet_resize(s_ab[0], trainer.patch_size,is_seg=True,axis=0) if s_ab is not None else np.zeros(trainer.patch_size)
        s_ab = np.expand_dims(s_ab, axis=0)
        s_ana = nnUNet_resize(s_ana[0], trainer.patch_size,is_seg=True,axis=0) if s_ana is not None else np.zeros(trainer.patch_size)
        s_ana = np.expand_dims(s_ana, axis=0)
        s = np.concatenate((s_ana,s_ab),axis=0)

        if identifier in pred_report and len(pred_report[identifier]):
            continue

        if r is not None:
            regions, gt_reports = trainer.split_batch_report([r])
        else:
            regions, gt_reports = None, None
        
        do_tta = False

        region_features, region_direction_names  = trainer.predict_preprocessed_data_return_region_report(
            d, s, regions, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision, modal=modal, eval_mode=eval_mode)

        region_features = torch.tensor(np.array([item.cpu().detach().numpy() for item in region_features]), dtype=torch.float32).to(trainer.llm_model.device)

        output = trainer.llm_model.generate(
                region_features,
                max_length=300,
                num_beams=1,
                num_beam_groups=1,
                do_sample=False,
                num_return_sequences = 1,
                early_stopping=True
        )
        del region_features

        generated_sents_for_selected_regions = trainer.tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        reference_sents_for_selected_regions = []
        for a in gt_reports:
            reference_sents_for_selected_regions.extend(a)
        
        regions = regions[0] # because eval batch_size = 1, so here we get the first element, [report1, report2,..]
        gt_reports = gt_reports[0] # [[ans_list1],[ana_list2],...]

        global_idx=None
        for idx, ana_list in enumerate(regions):
            if ana_list == "global" or ana_list=="mask":
                global_idx = idx
                global_report = gt_reports[idx]
        
        if eval_mode == "region_segtool":

            pred_global_report = generated_sents_for_selected_regions[-1]

            pred_region_concat_report = []
            for cur_idx, se in enumerate(generated_sents_for_selected_regions[:-1]):
                # no anatomy is mentioned in the sentence
                ana_flag = False
                for cur_a in hammer_anas:
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
                    # region_sentences[str(cur_idx)]={'report':se,"region":sort_ana}
                else:
                    if region_direction_names[cur_idx][0] == "left":
                        se = se.replace('right','left')
                    elif region_direction_names[cur_idx][0] == "right":
                        se = se.replace('left','right')
                    # region_sentences[str(cur_idx)]={'report':se,"region":region_direction_names[cur_idx][0]}
                pred_region_concat_report.append(se)

            pred_region_concat_report = " ".join(pred_region_concat_report)

            pred_split = pred_global_report.split('.')
            pred_split_2 = []
            for se in pred_split:
                pred_split_2.extend(se.split(','))
            pred_split = list(map(lambda x:x+'.',pred_split_2))
            
            if 'ventricle' not in pred_region_concat_report.lower():
                pred_region_concat_report = pred_region_concat_report+" "+" ".join([g for g in pred_split if 'ventricle' in g.lower()])            
            if 'midline' not in pred_region_concat_report.lower():
                pred_region_concat_report = pred_region_concat_report+" "+" ".join([g for g in pred_split if 'midline' in g.lower()])
            if 'sulci' not in pred_region_concat_report.lower():
                pred_region_concat_report = pred_region_concat_report+" "+" ".join([g for g in pred_split if 'sulci' in g.lower()])
            
            if 'midline' not in pred_region_concat_report.lower():
                pred_region_concat_report += " No midline shift."
            
            rouges, bleus = compute_language_model_scores([pred_region_concat_report, pred_global_report],[global_report, global_report])

            # pred_report[identifier]=[{"global_report":global_report,"pred_region_concat":{"report":pred_region_concat_report,"rouge":rouges[0],"bleu":bleus[0],"bert":bert_sims[0]}}]
            pred_report[identifier]=[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[1],"bleu":bleus[1]},"pred_region_concat":{"report":pred_region_concat_report,"rouge":rouges[0],"bleu":bleus[0]}}]
        
        elif eval_mode == "global":
            pred_global_report = generated_sents_for_selected_regions[global_idx]
            rouges, bleus = compute_language_model_scores([pred_global_report],[global_report])

            pred_oracle_region_reports = [g for idx,g in enumerate(generated_sents_for_selected_regions) if idx!=global_idx]
            gt_oracle_region_reports = [g for idx,g in enumerate(reference_sents_for_selected_regions) if idx!=global_idx]

            region_rouges, region_blues = compute_language_model_scores(pred_oracle_region_reports,gt_oracle_region_reports)

            # pred_report[identifier]=[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[0],"bleu":bleus[0],"bert":bert[0]}}]
            pred_report[identifier]=[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[0],"bleu":bleus[0]}}] + [{"gt_region":gt_pred[0],"pred_region":gt_pred[1],"rouge":gt_pred[2],"bleu":gt_pred[3]} for gt_pred in zip(gt_oracle_region_reports, pred_oracle_region_reports, region_rouges, region_blues)]

        elif eval_mode == "region_oracle":
            pred_global_report = generated_sents_for_selected_regions[global_idx]
            pred_region_concat_report = " ".join([g for idx,g in enumerate(generated_sents_for_selected_regions) if idx!=global_idx] + [g for g in pred_global_report.split('.') if 'midline' in g.lower() or 'sulci' in g.lower()])
            pred_oracle_region_reports = [g for idx,g in enumerate(generated_sents_for_selected_regions) if idx!=global_idx]
            gt_oracle_region_reports = [g for idx,g in enumerate(reference_sents_for_selected_regions) if idx!=global_idx]

            rouges, bleus = compute_language_model_scores([pred_region_concat_report, pred_global_report],[global_report, global_report])

            region_rouges, region_blues = compute_language_model_scores(pred_oracle_region_reports,gt_oracle_region_reports)
            
            # pred_report[identifier] =[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[1],"bleu":bleus[1],"bert":bert_sims[1]},"pred_region_concat":{"report":pred_region_concat_report,"rouge":rouges[0],"bleu":bleus[0],"bert":bert_sims[0]}}] + [{"gt":gt_pred[0],"pred":gt_pred[1]} for gt_pred in zip(gt_oracle_region_reports, pred_oracle_region_reports)]
            pred_report[identifier] =[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[1],"bleu":bleus[1]},"pred_region_concat":{"report":pred_region_concat_report,"rouge":rouges[0],"bleu":bleus[0]}}] + [{"gt_region":gt_pred[0],"pred_region":gt_pred[1],"rouge":gt_pred[2],"bleu":gt_pred[3]} for gt_pred in zip(gt_oracle_region_reports, pred_oracle_region_reports, region_rouges, region_blues)]
        elif eval_mode == "ft_region_oracle":
            pred_global_report = generated_sents_for_selected_regions[global_idx]
            rouges, bleus = compute_language_model_scores([pred_global_report],[global_report])
            pred_report[identifier] =[{"global_report":global_report,"pred_global_report":{"report":pred_global_report,"rouge":rouges[0],"bleu":bleus[0]}}]
        elif eval_mode == "given_mask":
            pred_region_concat_report = " ".join(generated_sents_for_selected_regions)
            rouges, bleus = compute_language_model_scores([pred_region_concat_report],[global_report])
            pred_report[identifier] =[{"global_report":global_report,"pred_report":{"report":pred_region_concat_report,"rouge":rouges[0],"bleu":bleus[0]}}]

        json_str = json.dumps(pred_report, indent=4)
        with open(os.path.join(output_folder,'pred_report.json'), 'w') as json_file:
            json_file.write(json_str)
        
        avg = {'pred_global_report':{'rouge':[],'bleu':[],'bert':[]},'pred_region_concat':{'rouge':[],'bleu':[]},'region':{'rouge':[],'bleu':[]}}

        for item in pred_report:
            if item == "avg":
                continue
            for ele in pred_report[item]:
                if 'pred_global_report' in ele.keys() or 'pred_region_concat' in ele.keys():
                    for ti in ele:
                        if ti == "pred_global_report" or ti == "pred_region_concat":
                            avg[ti]['rouge'].append(ele[ti]["rouge"])
                            avg[ti]['bleu'].append(ele[ti]["bleu"])
                elif 'pred_region' in ele.keys():
                    avg['region']['rouge'].append(ele['rouge'])
                    avg['region']['bleu'].append(ele['bleu'])

        # for item in ['bleu','rouge','bert']:
        for item in ['bleu','rouge']:
            for it in ['pred_global_report','pred_region_concat','region']:
                avg[it][item] = np.mean(avg[it][item]) if len(avg[it][item]) else ''
        pred_report['avg'] = avg
        
        json_str = json.dumps(pred_report, indent=4)
        with open(os.path.join(output_folder,'pred_report.json'), 'w') as json_file:
            json_file.write(json_str)

    print("inference done.")

    pool.close()
    pool.join()

    return 0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def predict_from_folder(model: str, seg_pretrained: str, output_folder: str, test_file: str,
                        num_threads_preprocessing: int, num_threads_nifti_save: int,
                        part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
                        overwrite_all_in_gpu: bool = None,
                        step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                        eval_mode="region_oracle"):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param output_folder:
    :param folds:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """

    setup_seed(42)

    maybe_mkdir_p(output_folder)

    test_file = json.load(open(test_file,'r'))
    list_of_lists = [[j['image']] for j in test_file]
    list_of_ab_segs = [j['label'] for j in test_file] if 'label' in test_file[0] else None
    list_of_ana_segs = [j['label2'] for j in test_file] if 'label2' in test_file[0] else None
    list_of_reports = [j['report'] for j in test_file] if 'report' in test_file[0] else None
    modals = [j['modal'] for j in test_file]

    case_identifiers = []
    for idx,j in enumerate(test_file):
        if 'wmh' in j['image']:
            case_identifiers.append(j.split('/')[-2]+'_'+j.split('/')[-1])
        elif 'Radio' in j['image'] or 'radio' in j['image']:
            case_identifiers.append(j['image'].split('/')[-3]+'_'+j['image'].split('/')[-2]+'_'+j['image'].split('/')[-1])
        elif 'myDWI' in j['image']:
            case_identifiers.append(j['image'].split('/')[-2]+'_'+j['image'].split('/')[-1].split('.')[0])
        else:
            case_identifiers.append(j['image'].split('/')[-1].split('.')[0])

    ### If the test process is continued from a prior test
    ### The following will automatically filter the untested samples and ignore the tested samples 
    if os.path.exists(os.path.join(output_folder,'pred_report.json')):

        out_report = json.load(open(os.path.join(output_folder,'pred_report.json'),'r')) 
        left_list_of_lists = []
        left_list_of_ab_segs = [] if list_of_ab_segs is not None else None
        left_list_of_ana_segs = [] if list_of_ana_segs is not None else None
        left_list_of_reports = [] if list_of_reports is not None else None 
        left_modals = []

        for idx in range(len(list_of_lists)):

            path = list_of_lists[idx][0]

            if 'wmh' in path:
                identifier = path.split('/')[-2]+'_'+path.split('/')[-1]
            elif 'Radio' in path or 'radio' in path:
                identifier = path.split('/')[-3]+'_'+path.split('/')[-2]+'_'+path.split('/')[-1]
            elif 'myDWI' in path:
                identifier = path.split('/')[-2] + '_' + path.split('/')[-1].split('.')[0]
            else:
                identifier = path.split('/')[-1].split('.')[0]
            
            if identifier in out_report and len(out_report[identifier]):
                continue

            left_list_of_lists.append(list_of_lists[idx])
            left_modals.append(modals[idx])

            if list_of_ab_segs is not None:
                left_list_of_ab_segs.append(list_of_ab_segs[idx])
            if list_of_ana_segs is not None:
                left_list_of_ana_segs.append(list_of_ana_segs[idx])
            if left_list_of_reports is not None:
                left_list_of_reports.append(list_of_reports[idx])
        
        list_of_lists = left_list_of_lists
        list_of_ab_segs = left_list_of_ab_segs
        list_of_ana_segs = left_list_of_ana_segs
        list_of_reports = left_list_of_reports
        modals = left_modals

        print("left",len(list_of_lists),len(list_of_reports))
    
    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu
    
    the_ab_segs = list_of_ab_segs[part_id::num_parts] if list_of_ab_segs is not None else None
    the_ana_segs = list_of_ana_segs[part_id::num_parts] if list_of_ana_segs is not None else None
    the_reports = list_of_reports[part_id::num_parts] if list_of_reports is not None else None

    return predict_cases(model, seg_pretrained, output_folder, list_of_lists[part_id::num_parts], the_ab_segs, the_ana_segs, the_reports, case_identifiers[part_id::num_parts], modals[part_id::num_parts],
                            num_threads_preprocessing, num_threads_nifti_save, tta,
                            mixed_precision=mixed_precision,
                            all_in_gpu=all_in_gpu,
                            step_size=step_size, checkpoint_name=checkpoint_name,
                            eval_mode=eval_mode)


if __name__ == "__main__":
    pass