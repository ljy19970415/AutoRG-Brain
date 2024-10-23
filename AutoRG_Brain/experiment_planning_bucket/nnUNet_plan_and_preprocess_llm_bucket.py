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

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from utilities.task_name_id_conversion import convert_id_to_task_name
from paths import *
from preprocess.sanity_checks_llm import verify_dataset_integrity
from experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from .utils_llm import crop

from .experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21

def get_npz_identifiers(cropped_out_dir):
    identifier = []
    for i in subfiles(cropped_out_dir, join=True, suffix=".pkl"):
        temp = i.split("/")[-1][:-4]
        if not temp.startswith('dataset_'):
            identifier.append(temp)
    return identifier

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder verify_dataset_integrity'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")
    
    ### use the segmentation model plan file to preprocess the report training data ####
    parser.add_argument("--plan_file", type=str, default="")


    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d

    if planner_name3d == "None":
        planner_name3d = None

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        args.verify_dataset_integrity = False

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))
        
        # crop do
        # 1 make crop output dir
        # 2 copy dataset.json to crop output dir
        # 3 copy label.nii.gz s to subdir "gt_segmentations" of preprocess out dir
        # 4 crop the data from images and labels from the dct list in "training" key in dataset.json
        # 4 and save them in nnUNet_raw/nnUNet_cropped_data/TaskXXX

        
        crop(task_name, False, tf)

        tasks.append(task_name)

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        bucket_cropped_out_dir = cropped_out_dir_bucket = nnUNet_cropped_data_bucket + "//" + t
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        bucket_preprocessing_output_dir_this_task = preprocessing_output_dir_bucket + "//" + t

        #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False

        # DatasetAnalyzer only use .pkl (which is very small, around 230KB)
        # so the pkl files are not on bucket
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        dataset_analyzer.patient_identifiers = get_npz_identifiers(cropped_out_dir)

        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        # if args.overwrite_plans is not None:
        #     assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
        #     exp_planner = ExperimentPlanner3D_v21(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
        #                                 args.overwrite_plans_identifier)
        # else:
        exp_planner = ExperimentPlanner3D_v21(cropped_out_dir, preprocessing_output_dir_this_task, bucket_cropped_out_dir, bucket_preprocessing_output_dir_this_task)
        
        if len(args.plan_file):
            exp_planner.plans = load_pickle(args.plan_file)
        else:
            exp_planner.plan_experiment()
        
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)

if __name__ == "__main__":
    main()

