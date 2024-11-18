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
from batchgenerators.utilities.file_and_folder_operations import *
from run.default_configuration import get_default_configuration
from paths import default_plans_identifier
from run.load_pretrained_weights import *
from network_training.nnUNetTrainerV2_six_pub_seg import nnUNetTrainerV2
from utilities.task_name_id_conversion import convert_id_to_task_name
from paths import preprocessing_output_dir_bucket
## to load test imgs and segs ##
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    
    #### use test_file to validate the model ####
    parser.add_argument("-train", "--train_file", type=str, required=False, default=None, help="use this if you want to train on customized ids")
    parser.add_argument("--only_ana", required=False, default=False, action="store_true",help="only optimize anatomy segmentation loss")
    parser.add_argument("--abnormal_type", type=str, required=False, default="intense",help="set the way yo synthesis abnormaly")
    parser.add_argument("--network_type", type=str, required=False, default="share",help="set the way yo synthesis abnormaly")

    parser.add_argument("-train_batch", "--num_batches_per_epoch", type=int, required=False, default=250)
    parser.add_argument("-val_batch", "--num_val_batches_per_epoch", type=int, required=False, default=50)

    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument("--disable_validation_inference", required=False, action="store_true",
                        help="If set nnU-Net will not run inference on the validation set. This is useful if you are "
                             "only interested in the test set results and want to save some disk space and time.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument("--anatomy_reverse", required=False, default=False, action="store_true")

    parser.add_argument("--bucket", help="whether the data is stored on s3", action="store_true")

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    train_file = args.train_file
    only_ana = args.only_ana
    abnormal_type = args.abnormal_type

    num_batches_per_epoch = args.num_batches_per_epoch
    num_val_batches_per_epoch = args.num_val_batches_per_epoch

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    network_type = args.network_type

    val_folder = args.val_folder

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage = get_default_configuration(network, task, network_trainer, plans_identifier)

    dataset_directory_bucket = preprocessing_output_dir_bucket+'//'+task if args.bucket else None
    
    if args.bucket:
        from petrel_client.client import Client
        client = Client('~/petreloss.conf') # client搭建了和ceph通信的通道
    else:
        client = None
    
    # print("plans_file",plans_file, fold, test_file, output_folder_name, dataset_directory, batch_dice, stage, decompress_data, deterministic, run_mixed_precision)

    # plans_file, fold, train_file, only_ana=False, abnormal_type="intense", num_batches_per_epoch=250, num_val_batches_per_epoch=50, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
    # unpack_data=True, deterministic=True, fp16=False, network_type="normal",dataset_directory_bucket=None,anatomy_reverse=False
    trainer = nnUNetTrainerV2(plans_file, fold, train_file, only_ana=only_ana, abnormal_type=abnormal_type, num_batches_per_epoch=num_batches_per_epoch, num_val_batches_per_epoch=num_val_batches_per_epoch, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,fp16=run_mixed_precision, 
                            network_type=network_type,dataset_directory_bucket=dataset_directory_bucket,anatomy_reverse=args.anatomy_reverse)
    trainer.client = client
    
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            # if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
            if args.pretrained_weights is not None:
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights_allow_missing(trainer.network, args.pretrained_weights)
            
            print("load latest checkpoint")
            trainer.load_latest_checkpoint()

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)
        
        # if test_file is not None:
        #     trainer.network.eval()
        #     dice_avg = trainer.validate()
        #     print("validation dice",dice_avg)
        

        # if args.disable_validation_inference:
        #     print("Validation inference was disabled. Not running inference on validation set.")
        # else:
        #     # predict validation

        # trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
        #                 run_postprocessing_on_folds=not disable_postprocessing_on_folds,
        #                 overwrite=args.val_disable_overwrite)

        # if network == '3d_lowres' and not args.disable_next_stage_pred:
        #     print("predicting segmentations for the next stage of the cascade")
        #     predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))

if __name__ == "__main__":
    main()