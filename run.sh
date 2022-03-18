#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=UTF-8



SET=$(seq 1 1 1)
for i in $SET

do

  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 trainSynth_concat.py --yaml="syn_concat_train" \



done

