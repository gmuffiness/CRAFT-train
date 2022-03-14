# sed -i -e 's/\r$//' run_fgh.sh
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 trainIC15_ddp.py --yaml=$EXP_NAME --port=2348
EXP_NAME=ic15_weak_supervision_shwang_test11_1_5_segment_region_score
yaml_path="config/$EXP_NAME.yaml"
cp config/ic15_train.yaml $yaml_path
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainIC15_ddp.py --yaml=$EXP_NAME --port=2351
rm "config/$EXP_NAME.yaml"