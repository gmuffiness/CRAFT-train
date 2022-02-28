"""
time: 33 sec | dataset size: 858750

"""

import sys
import time
import yaml

import torch

from config.load_config import cfg
from data.dataset import SynthTextDataSet
from utils.util import saveInput, saveImage

if __name__ == "__main__":
    # print(sys.path)
    start_time = time.time()
    config = cfg
    print(yaml.dump(config.train.data))

    synth_dataset = SynthTextDataSet(
        output_size=config.train.data.output_size,
        data_dir=config.data_dir.synthtext,
        saved_gt_dir=config.data_dir.synthtext_gt,
        gauss_init_size=config.train.data.gauss_init_size,
        gauss_sigma=config.train.data.gauss_sigma,
        enlarge_size=config.train.data.enlarge_size,
        aug=config.train.data.aug,
        logging=config.train.data.logging,
    )

    synth_train_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    batch_synth = iter(synth_train_loader)
    synth_images, synth_region_label, synth_affi_label, synth_confidence_mask = next(
        batch_synth
    )
    # print(synth_images.shape)
    if config.train.data.logging:
        saveInput(
            'test_img',
            synth_images,
            synth_region_label,
            synth_affi_label,
            synth_confidence_mask,
        )

    print(f"elapsed time : {time.time() - start_time}")
