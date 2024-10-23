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

from network_training.nnUNetTrainerV2_llm_resize_new import nnUNetTrainerV2 as nnUNetTrainerV2_resize
from utilities.task_name_id_conversion import convert_id_to_task_name

from paths import preprocessing_output_dir_bucket
# from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
# from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
# from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes

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

    parser.add_argument("--plans_file", type=str, required=False,help="the plans_file of the segmentation module")

    parser.add_argument("-train_batch", "--num_batches_per_epoch", type=int, required=False, default=250)
    parser.add_argument("-val_batch", "--num_val_batches_per_epoch", type=int, required=False, default=50)

    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--no_aug", help="use this if you don't want any augmentation", action="store_true")
    parser.add_argument("--bucket", help="whether the data is stored on s3", action="store_true")
    
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
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-seg_pretrained', type=str, required=True, default=None,
                        help='path to nnU-Net checkpoint file to be used as segmentation (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument('--finetune_from',type=str,default=None)
    parser.add_argument('--train_with_seg', action='store_true', default=False)
    
    parser.add_argument("--feature_layer", type=int, required=False, default=2)
    parser.add_argument("--avg_type", type=str, required=False, default='xyz')
    parser.add_argument("--pool_to_feature_layer", type=int, required=False, default=None)
    parser.add_argument('--use_conv_pool', action='store_true', default=False)
    parser.add_argument('--use_patchwise', action='store_true', default=False)

    ## size = 4, the input patch size of llm will be 4*4*4
    ## size = 8, the input patch size of llm will be 8*8*8
    parser.add_argument("--size", type=int, required=False, default=4)

    parser.add_argument("--max_tokens", type=int, required=False, default=1024)

    parser.add_argument("--dataset", type=str, required=False, default="six")

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

    train_with_seg = args.train_with_seg

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage = get_default_configuration(network, task, network_trainer, plans_identifier, plans_file = args.plans_file)

    dataset_directory_bucket = preprocessing_output_dir_bucket+'//'+task if args.bucket else None

    if args.bucket:
        from petrel_client.client import Client
        client = Client('~/petreloss.conf') # client搭建了和ceph通信的通道
    else:
        client = None
    
    trainer = nnUNetTrainerV2_resize(plans_file, fold, train_file, only_ana=only_ana, abnormal_type=abnormal_type, num_batches_per_epoch=num_batches_per_epoch, num_val_batches_per_epoch=num_val_batches_per_epoch, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision, network_type=network_type,feature_layer=args.feature_layer,dataset_directory_bucket=dataset_directory_bucket,train_with_seg=train_with_seg,avg_type=args.avg_type,pool_to_feature_layer=args.pool_to_feature_layer,use_conv_pool=args.use_conv_pool,
                            use_global = args.use_patchwise, size=args.size, dataset=args.dataset, max_tokens=args.max_tokens)
    trainer.client = client
    
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only, no_aug = args.no_aug)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.finetune_from is not None:
                saved_model = torch.load(args.finetune_from, map_location=torch.device('cpu'))
                trainer.llm_model.load_state_dict(saved_model['state_dict'])
                print("!!!!!!!!! load finetune from",args.finetune_from)
            # if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                # continur learning for llm model
            # trainer.load_latest_checkpoint()
            
            ### load segmentation pretrained weights ###
            load_pretrained_weights(trainer.network, args.seg_pretrained)

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

if __name__ == "__main__":
    main()