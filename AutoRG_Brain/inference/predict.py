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


import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import *
import sys
if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue
import torch
import shutil
from multiprocessing import Pool

from inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from network_training.model_restore import load_model_and_checkpoint_files
from utilities.nd_softmax import *

#from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, list_of_segs, output_files, transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            the_seg = list_of_segs[i] if list_of_segs is not None else None
            if 'Radio_VQA' in l[0] or 'radio' in l[0]:
                target_shape = (24,320,320)
            else:
                target_shape = None
            
            d, s, dct = preprocess_fn(l, the_seg, target_shape=target_shape)
            # print(output_file, dct)
            
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
    # restore output
    # sys.stdout = sys.__stdout__


def preprocess_multithreaded(trainer, list_of_lists, list_of_segs, output_files, num_processes=2):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    # classes = list(range(1, trainer.num_classes)) # 96

    # assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []

    for i in range(num_processes):
        the_segs = list_of_segs[i::num_processes] if list_of_segs is not None else None
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes], the_segs,
                                                            output_files[i::num_processes], trainer.plans['transpose_forward']))
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

def predict_cases(model, output_folder, list_of_lists, list_of_segs, output_filenames, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, dice_type="anatomy", do_tta=True, save_output_nii=False,
                  mixed_precision=True,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False, modal=None):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """

    # assert len(list_of_lists) == len(output_filenames) and len(list_of_lists) == len(list_of_segs)

    # num_threads_nifti_save, 2
    
    pool = Pool(num_threads_nifti_save)
    results = []

    # cleaned_output_files is a list of the path of output_prediction.nii.gz
    # e.g ['./inferTs/BraTS20_Training_001_flair.nii.gz',...]
    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))
    
    # below code won't run
    # if not overwrite_existing:
    #     print("number of cases:", len(list_of_lists))
    #     # if save_npz=True then we should also check for missing npz files
    #     not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

    #     cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
    #     list_of_lists = [list_of_lists[i] for i in not_done_idx]
    #     if segs_from_prev_stage is not None:
    #         segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

    #     print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    # if folds is a int, e.g. 0, then len(params) == 1
    trainer, params = load_model_and_checkpoint_files(model, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    
    if 'Radio_VQA' in list_of_lists[0][0] or 'radio' in list_of_lists[0][0]:
        trainer.plans['transpose_forward'] = [2,0,1]
        trainer.transpose_forward = [2,0,1]
        trainer.plans['transpose_backward'] = [1,2,0]
        trainer.transpose_backward = [1,2,0]

    if segmentation_export_kwargs is None: # choose this
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    # under 3dfullres setting, seg_from_prev_stage is None
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, list_of_segs, cleaned_output_files, num_threads_preprocessing)
    print("starting prediction...")
    # this d is from trainer.preprocess_patient

    print("len list of lists",len(list_of_lists),len(list_of_segs))

    if list_of_segs is not None:
        dices = []

    if os.path.exists(os.path.join(output_folder,'test_dices.json')):
        test_dices = json.load(open(os.path.join(output_folder,'test_dices.json'),'r'))
    else:
        test_dices = {}
    
    reverse_anatomy_map = json.load(open('utils_file/hammer_label_reverse_map.json','r'))

    for preprocessed in preprocessing:

        output_filename, (d, s, dct) = preprocessed

        if save_output_nii:
            if os.path.exists(output_filename):
                continue
        elif output_filename.split('/')[-1] in test_dices:
            continue
        
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        # load the params of the network
        trainer.load_checkpoint_ram(params[0], False)
        
        do_tta = False

        softmaxs = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision, modal=modal)
        
        softmax_abnormal, softmax_anatomy = softmaxs[1], softmaxs[3]
        
        # if fold is a int then len(params) == 1, the below code won't run
        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision, modal=modal)[1]
        # if fold is a int then len(params) == 1, the below code won't run
        if len(params) > 1:
            softmax /= len(params)

        # softmax_transpose.shape = num_classes, 144, 174, 138
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_abnormal = softmax_abnormal.transpose([0] + [i + 1 for i in transpose_backward]) # 2, x ,y z
            softmax_anatomy = softmax_anatomy.transpose([0] + [i + 1 for i in transpose_backward]) # 96, x, y, z
        
        if list_of_segs is not None:
            gt = s[0]
            gt[gt<0] = 0

            if "anatomy" in dice_type:
                ###### begin annotation ####
                class_dice = []
                pred = softmax_anatomy.argmax(0)
                if dice_type == "anatomy_reverse":
                    pred_reverse = np.zeros(pred.shape)
                    for from_label, to_label in reverse_anatomy_map.items():
                        pred_reverse[pred == int(from_label)] = int(to_label)
                    
                    pred = pred_reverse

                for ana in list(np.unique(gt)):
                    if ana == 0:
                        continue

                    hd = cal_hd(pred==ana,gt==ana)
                    # nsd = cal_nsd(pred==ana,gt==ana)
                    cur_dice, nsd, precision, sensitivity, specificity = cal_dice(pred==ana,gt==ana)

                    class_dice.append([cur_dice, nsd, hd, precision, sensitivity, specificity])

                    # class_dice.append(cal_dice(pred==ana, gt==ana))
                    
                dices.append(np.mean(class_dice,axis=0))

                ### end annotation ####
            else:
                pred = softmax_abnormal.argmax(0)
                gt[gt>0] = 1
                gt[gt<0] = 0
                hd = cal_hd(pred,gt)
                cur_dice, nsd, precision, sensitivity, specificity = cal_dice(pred,gt)
                cur_dice = 0 if np.isnan(cur_dice) else cur_dice
                nsd = 0 if np.isnan(nsd) else nsd
                precision = 0 if np.isnan(precision) else precision
                sensitivity = 0 if np.isnan(sensitivity) else sensitivity
                specificity = 0 if np.isnan(specificity) else specificity
                dices.append([cur_dice, nsd, hd, precision, sensitivity, specificity])
        
            test_dices[output_filename.split('/')[-1]] = {'dice':dices[-1][0],'nsd':dices[-1][1],'hd':dices[-1][2],'precision':dices[-1][3],'sensitivity':dices[-1][4],'specificity':dices[-1][5]}

            test_dices["avg"] = {'dice':np.mean([test_dices[i]['dice'] for i in test_dices if not i.startswith("avg")]),
            'nsd':np.mean([test_dices[i]['nsd'] for i in test_dices if not i.startswith("avg")]),
            'hd':np.mean([test_dices[i]['hd'] for i in test_dices if not i.startswith("avg")]),
            'precision':np.mean([test_dices[i]['precision'] for i in test_dices if not i.startswith("avg")]),
            'sensitivity':np.mean([test_dices[i]['sensitivity'] for i in test_dices if not i.startswith("avg")]),
            'specificity':np.mean([test_dices[i]['specificity'] for i in test_dices if not i.startswith("avg")])}

            json_str = json.dumps(test_dices, indent=4)
            with open(os.path.join(output_folder,'test_dices.json'), 'w') as json_file:
                json_file.write(json_str)

        # pred = softmax.argmax(0) # shape x,y,z
        # print("softmax transpose",pred.shape,list(np.unique(pred)),"d",d.shape,"gt",gt.shape,list(np.unique(gt)))

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        if save_output_nii:

            # if "abnormal" in dice_type:
            output_filename_ab = output_filename[:-7] +"_ab_mask.nii.gz"
            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                            ((softmax_abnormal, output_filename_ab, dct, interpolation_order, region_class_order,
                                                None, None,
                                                npz_file, None, force_separate_z, interpolation_order_z),)
                                            ))
            # if "anatomy" in dice_type:
            output_filename_ana = output_filename[:-7] +"_ana_mask.nii.gz"
            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                            ((softmax_anatomy, output_filename_ana, dct, interpolation_order, region_class_order,
                                                None, None,
                                                npz_file, None, force_separate_z, interpolation_order_z, True, dice_type=="anatomy_reverse"),)
                                            ))

            if list_of_segs is None and dice_type != "report":
                test_dices[output_filename.split('/')[-1]] = output_filename_ana
                save_json(test_dices, join(output_folder,'test_dices.json'))

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    if not disable_postprocessing:
        results = []
        pp_file = join(model, "postprocessing.json")
        if isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
            # for_which_classes stores for which of the classes everything but the largest connected component needs to be
            # removed
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(pool.starmap_async(load_remove_save,
                                              zip(output_filenames, output_filenames,
                                                  [for_which_classes] * len(output_filenames),
                                                  [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                  "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                  "%s" % model)

    pool.close()
    pool.join()

    if list_of_segs is not None:
        # return test_dices['avg']
        return 0
    else:
        return 0

def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    #### test ###
    if True:
        maybe_case_ids = np.unique([i.split('.')[0] for i in files])
        return maybe_case_ids
    #### test ###

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids

def predict_from_folder(model: str, output_folder: str, test_file: str,
                        save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                        dice_type: str,
                        part_id: int, num_parts: int, tta: bool, save_output_nii: bool, mixed_precision: bool = True,
                        overwrite_all_in_gpu: bool = None,
                        step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                        segmentation_export_kwargs: dict = None, disable_postprocessing: bool = True, modal=None):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :return:
    """

    maybe_mkdir_p(output_folder)
    # shutil.copy(join(model, 'plans.pkl'), output_folder)
    # assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    # expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    test_file = json.load(open(test_file,'r'))
    list_of_lists = [[j['image']] for j in test_file]
    list_of_segs = [j['label'] for j in test_file] if 'label' in test_file[0] else None


    output_folders = []
    for j in test_file:
        if 'modal' in j and 'dis' in j:
            output_folders.append(join(output_folder,j['dis'],j['modal']))
        elif 'modal' in j:
            output_folders.append(join(output_folder,j['modal']))
        elif 'dis' in j:
            output_folders.append(join(output_folder,j['dis']))
        else:
            output_folders.append(output_folder)
    
    output_files = []
    # get the unique identifier for each test sample
    for idx,j in enumerate(test_file):
        if 'WMH_Segmentation_Challenge' in j['image']:
            output_files.append(join(output_folders[idx], j['image'].split('/')[-4]+'_'+j['image'].split('/')[-3]+'_'+j['image'].split('/')[-1]))
        elif 'Radio' in j['image'] or 'radio' in j['image']:
            output_files.append(join(output_folders[idx], j['image'].split('/')[-3]+'_'+j['image'].split('/')[-2]+'_'+j['image'].split('/')[-1]))
        elif 'myDWI' in j['image']:
            output_files.append(join(output_folders[idx], j['image'].split('/')[-2]+'_'+j['image'].split('/')[-1]))
        elif list_of_segs is None:
            output_files.append(join(output_folders[idx], j['image'].split('/')[-1]))
        else:
            output_files.append(join(output_folders[idx], j['image'].split('/')[-1]))
    
    print("before",len(list_of_lists),len(output_files))

    assert len(output_files) == len(list_of_lists)

    if list_of_segs is not None:
        assert len(output_files) == len(list_of_segs)

    if save_output_nii:

        left_output_files, left_list_of_lists = [], []
        left_list_of_segs = [] if list_of_segs is not None else None

        for idx in range(len(output_files)):

            output_filename = output_files[idx]
            output_filename_ab = output_filename[:-7] +"_ab_mask.nii.gz"
            output_filename_ana = output_filename[:-7] +"_ana_mask.nii.gz"

            if os.path.exists(output_filename_ab) and os.path.exists(output_filename_ana):
                continue

            # if (dice_type=="abnormal" and os.path.exists(output_filename_ab)) or ("anatomy" in dice_type and os.path.exists(output_filename_ana)):
            #     continue
            
            left_output_files.append(output_files[idx])
            left_list_of_lists.append(list_of_lists[idx])

            if list_of_segs is not None:
                left_list_of_segs.append(list_of_segs[idx])
    
    print("left",len(left_list_of_lists),len(left_output_files))

    if len(left_list_of_lists) == 0:
        print("all prediction done already")
        return

    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu
    
    the_segs = left_list_of_segs[part_id::num_parts] if list_of_segs is not None else None

    return predict_cases(model, output_folder, left_list_of_lists[part_id::num_parts], the_segs, left_output_files[part_id::num_parts],
                            save_npz, num_threads_preprocessing, num_threads_nifti_save, dice_type, tta,
                            save_output_nii = save_output_nii,
                            mixed_precision=mixed_precision,
                            all_in_gpu=all_in_gpu,
                            step_size=step_size, checkpoint_name=checkpoint_name,
                            segmentation_export_kwargs=segmentation_export_kwargs,
                            disable_postprocessing=disable_postprocessing, modal=modal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-m', '--model_output_folder',
                        help='model output folder. Will automatically discover the folds '
                             'that were '
                             'run and use those as an ensemble', required=True)
    parser.add_argument('-f', '--folds', nargs='+', default='None', help="folds to use for prediction. Default is None "
                                                                         "which means that folds will be detected "
                                                                         "automatically in the model output folder")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="use this if you want to ensemble"
                                                                                      " these predictions with those of"
                                                                                      " other models. Softmax "
                                                                                      "probabilities will be saved as "
                                                                                      "compresed numpy arrays in "
                                                                                      "output_folder and can be merged "
                                                                                      "between output_folders with "
                                                                                      "merge_predictions.py")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None', help="if model is the highres "
                                                                                             "stage of the cascade then you need to use -l to specify where the segmentations of the "
                                                                                             "corresponding lowres unet are. Here they are required to do a prediction")
    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you "
                             "want to use n GPUs to predict this "
                             "folder you need to run this command "
                             "n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different "
                             "GPU (via "
                             "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")
    parser.add_argument("--tta", required=False, type=int, default=1, help="Set to 0 to disable test time data "
                                                                           "augmentation (speedup of factor "
                                                                           "4(2D)/8(3D)), "
                                                                           "lower quality segmentations")
    parser.add_argument("--overwrite_existing", required=False, type=int, default=1, help="Set this to 0 if you need "
                                                                                          "to resume a previous "
                                                                                          "prediction. Default: 1 "
                                                                                          "(=existing segmentations "
                                                                                          "in output_folder will be "
                                                                                          "overwritten)")
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations, has no effect if mode=fastest")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z is z is done differently")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest")
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    model = args.model_output_folder
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    tta = args.tta
    step_size = args.step_size

    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    mode = args.mode
    all_in_gpu = args.all_in_gpu

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if overwrite == 0:
        overwrite = False
    elif overwrite == 1:
        overwrite = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    predict_from_folder(model, input_folder, output_folder, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, dice_type, part_id, num_parts, tta,
                        mixed_precision=not args.disable_mixed_precision,
                        mode=mode, overwrite_all_in_gpu=all_in_gpu, step_size=step_size)
