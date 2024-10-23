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

from inference.predict_llm import predict_from_folder
from paths import default_plans_identifier, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir, save_json
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('--model_folder', required=True, default=None, help="folder of saved segmentation model")
    parser.add_argument('-test', "--test_file", required=False,default=None, help="folder of groudtruth segmentation")
    parser.add_argument('-chk', help='checkpoint name, default: AutoRG_Brain_RGv1', default='AutoRG_Brain_RGv1')

    parser.add_argument('-seg_pretrained', type=str, required=True, default=None,
                        help='path to nnU-Net checkpoint file to be used as segmentation (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument("--eval_mode", type=str, default="global", choices=["ft_region_oracle","region_oracle","region_segtool","global","given_mask"])

    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)

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

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--overwrite_existing", required=False, default=True, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")

    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')
    
    args = parser.parse_args()
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = args.step_size
    all_in_gpu = args.all_in_gpu

    test_file = args.test_file

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False
    
    ### language model folder ###
    model_folder_name = args.model_folder
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    st = time()
    predict_from_folder(model_folder_name, args.seg_pretrained, output_folder, test_file, num_threads_preprocessing,
                        num_threads_nifti_save, part_id, num_parts, not disable_tta,
                        overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not args.disable_mixed_precision,
                        step_size=step_size, checkpoint_name=args.chk, eval_mode = args.eval_mode)
    end = time()
    save_json(end - st, join(output_folder, 'prediction_time.txt'))

if __name__ == "__main__":
    main()
