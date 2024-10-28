#!/bin/sh

#id="ucm_GeoRSCLIP_mul2_AwareTrans5Dec+4"
#id="GeoRSCLIP_mul2_AwareTrans5Dec+4"
#id="GeoRSCLIP_Trans5Dec"
id="GeoRSCLIP[cls]_Trans5Dec"
#id="demo"

#cfg="configs/transformer/transformer.yml"
cfg="configs/transformer/transformer.yml"

python train.py --cfg $cfg --id $id --start_from log_${id}

python eval.py  --dump_json 1 --dump_images 0 --num_images -1 --model log_${id}/model-best.pth --infos_path log_${id}/infos_${id}-best.pkl --language_eval 1 --beam_size 3

#beam_size为4,可能会高、、