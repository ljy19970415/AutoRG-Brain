# AutoRG_Brain_RGv1
srun -p partition_name -N 1 --quotatype auto --gres=gpu:1 python test_llm.py --eval_mode given_mask --model_folder /path/to/checkpoint/folder -chk AutoRG_Brain_RGv1 -seg_pretrained /path/to/saved/segmentation/checkpoint/AutoRG_Brain_SEG.model -test /path/to/the/above/test_file.json -o /output/folder/to/save/generated/report

# AutoRG_Brain_RGv2
srun -p partition_name -N 1 --quotatype auto --gres=gpu:1 python test_llm.py --eval_mode given_mask --model_folder /path/to/checkpoint/folder -chk AutoRG_Brain_RGv2 -seg_pretrained /path/to/saved/segmentation/checkpoint/AutoRG_Brain_SEG.model -test /path/to/the/above/test_file.json -o /output/folder/to/save/generated/report