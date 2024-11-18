from inference.inferenceSdk import AutoRG_Brain
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-o',"--out_dir", required=True, default=None, help="folder for saving mask and report predictions")
parser.add_argument('--llm_folder', required=True, default=None, help="folder of saved llm checkpoint")
parser.add_argument('--llm_chk', help='llm checkpoint name (if xx.model then --llm_chk xx)', default='AutoRG_Brain_RGv1')
parser.add_argument('--seg_folder', required=True, default=None, help="folder of saved segmentation checkpoint")
parser.add_argument('--seg_chk', help='segmentation checkpoint name (if xx.model then --seg_chk xx)', default='AutoRG_Brain_SEG')
parser.add_argument('-test','--test_file', required=True,default=None, help="json with your test images info")
parser.add_argument('--eval_mode', required=False,default='region_segtool', help="the report inference way")

args = parser.parse_args()

config = {
    'llm_folder':args.llm_folder,
    'seg_folder':args.seg_folder,
    'llm_chk':args.llm_chk,
    'seg_chk':args.seg_chk,
    'output_dir':args.out_dir,
    'eval_mode':args.eval_mode
}

model = AutoRG_Brain(gpu_id=[0], config=config)

input_case_dict = json.load(open(args.test_file,'r'))

results = model.report(input_case_dict)

with open(os.path.join(args.out_dir,'pred_report.json'), 'w') as f:
    json.dump(results, f, indent=4)