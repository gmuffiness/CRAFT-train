# CRAFT-Refactoring

```bash
├── data
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   ├── imgproc.py
│   └── preproc_synth.py
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── model
│   ├── craft.py
│   └── vgg16_bn.py
├── utils
│   ├── config.py
│   ├── craft_utils.py
│   ├── inference_boxes.py
│   └── utils.py
├── trainSynth.py
└── eval.py
```

### Training

1. Write yaml file   
2. Put the yaml file in the config folder  
3. Run trainSynth.py

```
CUDA_VISIBLE_DEVICES=0,1 python3 trainSynth.py --yaml=test
```
* ```--yaml:``` yaml file name
+ The experiment results will be saved to ```./exp/yaml``` by default.



