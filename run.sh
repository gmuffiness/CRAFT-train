# Guide : Set PYTHONPATH
# 1. vim ~/.bashrc
# Add below line to 'bashrc' file
# 2. export PYTHONPATH=$PYTHONPATH:${YOUR_PROJECT_ROOT_PATH}
#CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 python3 trainSynth.py --yaml=shwang_synthtext_test6_26
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 trainSynth.py --yaml=default

CUDA_VISIBLE_DEVICES=4,6 python3 trainIC15.py --yaml=ic15_test6_26