# CRAFT-train

```bash
├── config
│   ├── syn_train.yaml
│   ├── ic15_train.yaml
├── data
│   ├── pseudo_label
│   │   ├── make_charbox.py
│   │   └── watershed.py
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   ├── imgproc.py
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── model
│   ├── craft.py
│   └── vgg16_bn.py
├── utils
│   ├── craft_utils.py
│   ├── inference_boxes.py
│   └── utils.py
├── trainSynth.py
├── trainIC15.py
└── eval.py
```

### Training

1. Write configuration in yaml format
2. Put the yaml file in the config folder
3. Run train code (trainSynth.py or trainIC15.py)
4. Then, experiment results will be saved to ```./exp/[yaml]``` by default.

```
CUDA_VISIBLE_DEVICES=0,1 python3 trainSynth.py --yaml=syn_train   # run supervision code
CUDA_VISIBLE_DEVICES=0,1 python3 trainIC15.py --yaml=ic15_train   # run weak-supervision code
```

### Arguments
* ```--yaml``` : configuration file name

### Evaluation

| Training Dataset   | Evaluation Dataset   | Precision  | Recall  | F1-score  | pretrained model  |
| ------------- |-----|:-----:|:-----:|:-----:|-----:|
| SynthText      |  ICDAR2013 | 0.801 | 0.748 | 0.773| <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">download link</a>|
| SynthText + ICDAR2015      | ICDAR2015  | 0.909 | 0.794 | 0.848| <a href="https://drive.google.com/file/d/1qUeZIDSFCOuGS9yo8o0fi-zYHLEW6lBP/view">download link</a>|
