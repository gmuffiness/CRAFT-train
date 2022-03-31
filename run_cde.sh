EXP_NAME=debug_syn_train_base_en_ko_ai
yaml_path="config/$EXP_NAME.yaml"
cp config/syn_train_base_en_ko_ai.yaml $yaml_path
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainSynth.py --yaml=$EXP_NAME --port=2468
rm "config/$EXP_NAME.yaml"