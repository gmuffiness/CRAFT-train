# sed -i -e 's/\r$//' run_cde.sh
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 trainIC15_ddp.py --yaml=$EXP_NAME --port=2348
EXP_NAME=debug_syn_train_base_en_ko_ai
yaml_path="config/$EXP_NAME.yaml"
cp config/syn_train_base_en_ko_ai.yaml $yaml_path
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainSynth.py --yaml=$EXP_NAME --port=2468
rm "config/$EXP_NAME.yaml"