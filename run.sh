#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=UTF-8



SET=$(seq 1 1 1)
for i in $SET

do


  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 trainIC15_ddp.py --yaml="prescrip_train_v3"

done



#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainIC15_ddp.py --yaml="prescrip_train_time_test_before"

#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainIC15_ddp.py --yaml="test_test" \
#CUDA_VISIBLE_DEVICES=1 python3 eval_v2.py --yaml="eval" \


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainIC15_ddp.py --yaml="prescrip_vgg_test"