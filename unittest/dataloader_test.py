"""
time: 33 sec | dataset size: 858750

"""

import sys
import time

import torch

from config.load_config import cfg
from data.dataset import SynthTextDataLoader

if __name__ == "__main__":
    # print(sys.path)
    start_time = time.time()
    config = cfg

    synth_text_data_loader = SynthTextDataLoader(
        output_size=config.train.data.output_size,
        data_dir=config.data_dir.synthtext,
        saved_gt_dir=config.data_dir.synthtext_gt,
        logging=config.train.data.logging,
    )

    synth_train_loader = torch.utils.data.DataLoader(
        synth_text_data_loader,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    batch_synth = iter(synth_train_loader)
    synth_images, synth_region_label, synth_affi_label, synth_confidence_mask = next(
        batch_synth
    )
    print(synth_images.shape)
    print(f"elapsed time : {time.time() - start_time}")
