from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from paths import *
from experiment_planning_bucket.cropping_llm_bucket import ImageCropper
import shutil
from configuration import default_num_threads
import json

def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            # cur_pat.append(tr['image'][:-7] +"_%04.0d.nii.gz" % mod)
            cur_pat.append(tr['image'])
        cur_pat.append(tr['label1'])
        cur_pat.append(tr['label2'])
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def crop(task_string, override=False, num_threads=default_num_threads):

    cropped_out_dir_bucket = nnUNet_cropped_data_bucket + "//" + task_string
    preprocess_out_dir_bucket = preprocessing_output_dir_bucket + "//" + task_string

    cropped_out_dir = join(nnUNet_cropped_data, task_string)

    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    
    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # lists [[img_path,label_path1,label_path2],[img_path,label_path1,label_path2],...]


    imgcrop = ImageCropper(num_threads, cropped_out_dir, cropped_out_dir_bucket, preprocess_out_dir_bucket)
  
    
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)

