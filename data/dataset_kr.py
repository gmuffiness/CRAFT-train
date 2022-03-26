import os
import re
import itertools
import copy

import h5py
import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import (
    random_crop_with_bbox,
    random_horizontal_flip,
    random_rotate,
    random_scale,
    random_resize_crop,
    random_resize_crop_synth
)
from utils.util import saveInput, saveImage
from data.pseudo_label.make_charbox import PseudoCharBoxBuilder

def hierarchical_dataset(root, config, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []

    print(f'dataset_root:    {root}\t dataset: {select_data[0]}')
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        for i in filenames:
            lmdb_path = os.path.join(dirpath, str(i))
            dataset = SynthTextDataSet_kr(
                output_size=config.train.data.output_size,
                data_dir=lmdb_path,
                gauss_init_size=config.train.data.gauss_init_size,
                gauss_sigma=config.train.data.gauss_sigma,
                enlarge_region=config.train.data.enlarge_region,
                enlarge_affinity=config.train.data.enlarge_affinity,
                aug=config.train.data.syn_kor_aug,
                vis_opt=config.train.data.vis_opt,
            )

            print(f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}')
            dataset_list.append(dataset)

    return dataset_list



class SynthTextDataSet_kr(Dataset):
    def __init__(
        self,
        output_size,
        data_dir,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_opt,
    ):

        self.output_size = output_size
        self.data_dir = data_dir
        #self.gt = self.load_data(data_dir)


        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )
        self.aug = aug
        self.vis_opt = vis_opt

        self.gt = None
        with h5py.File(self.data_dir, 'r') as file:
            self.img_names = np.array(list(file['data'].keys()))



    # NOTE
    def load_data(self, path):

        folder, ext = os.path.splitext(path)
        if ext == '.h5':
            gt = h5py.File(path, 'r')
        else:
            gt = h5py.File(os.path.join(path, 'dset_kr.h5'), 'r')

        return gt

    @property
    def get_gt(self):
        if self.gt is None:
            self.gt = self.load_data(self.data_dir)
        return self.gt


    def make_pseudo_gt(self, index):

        gt = self.get_gt['data'][self.img_names[index]]

        image = gt[...] #RGB

        #print('img size : {}'.format(image.shape))

        charBB = gt.attrs['charBB']
        txt = gt.attrs['txt']


        all_char_bbox = charBB.transpose((2, 1, 0))
        image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        img_h, img_w, _ = image.shape
        confidence_mask = np.ones((img_h, img_w), dtype=np.uint8)

        try:
            words = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
        except:
            txt = [t.decode('UTF-8') for t in txt]
            words = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]

        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        word_level_char_bbox = []
        char_idx = 0
        for i in range(len(words)):
            length_of_word = len(words[i])
            word_bbox = all_char_bbox[char_idx: char_idx + length_of_word]
            assert len(word_bbox) == length_of_word
            char_idx += length_of_word
            word_bbox = np.array(word_bbox)
            word_level_char_bbox.append(word_bbox)


        region_score = self.gaussian_builder.generate_region(
            img_h, img_w, word_level_char_bbox, horizontal_text_bools=[True for _ in range(len(words))]
        )
        affinity_score, _ = self.gaussian_builder.generate_affinity(
            img_h, img_w, word_level_char_bbox, horizontal_text_bools=[True for _ in range(len(words))]
        )


        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        )

    def dilate_img_to_output_size(self, image, char_bbox):
        h, w = image.shape[0:2]
        if min(h, w) <= self.output_size:
            scale = float(self.output_size + 10) / min(h, w)
        else:
            scale = 1.0
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        char_bbox *= scale
        return image, char_bbox

    def augment_image(
            self, image, region_score, affinity_score, confidence_mask, word_level_char_bbox
    ):

        augment_targets = [image, region_score, affinity_score, confidence_mask]

        if self.aug.random_scale.option:
            augment_targets, word_level_char_bbox = random_scale(
                augment_targets, word_level_char_bbox, self.aug.random_scale.range
            )

        if self.aug.random_rotate.option:
            augment_targets = random_rotate(
                augment_targets, self.aug.random_rotate.max_angle
            )

        if self.aug.random_crop.option:
            if (
                    self.aug.random_crop.version
                    == "random_crop_with_bbox"
            ):
                augment_targets = random_crop_with_bbox(
                    augment_targets, word_level_char_bbox, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop":
                augment_targets = random_resize_crop(
                    augment_targets,
                    self.aug.random_crop.scale,
                    self.aug.random_crop.ratio,
                    self.output_size,
                    self.aug.random_crop.rnd_threshold
            )
            elif self.aug.random_crop.version == "random_resize_crop_synth":
                augment_targets = random_resize_crop_synth(
                    augment_targets, self.output_size
                )

            else:
                assert "Undefined RandomCrop version"

        if self.aug.random_horizontal_flip.option:
            augment_targets = random_horizontal_flip(augment_targets)

        if self.aug.random_colorjitter.option:
            image, region_score, affinity_score, confidence_mask = augment_targets
            image = Image.fromarray(image)
            image = transforms.ColorJitter(
                brightness=self.aug.random_colorjitter.brightness,
                contrast=self.aug.random_colorjitter.contrast,
                saturation=self.aug.random_colorjitter.saturation,
                hue=self.aug.random_colorjitter.hue,
            )(image)
        else:
            image, region_score, affinity_score, confidence_mask = augment_targets

        return np.array(image), region_score, affinity_score, confidence_mask

    def resize_to_half(self, ground_truth):
        return cv2.resize(ground_truth, (self.output_size // 2, self.output_size // 2))

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, index):

        (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        ) = self.make_pseudo_gt(index)


        # if self.logging:
        #     saveImage(self.img_names[index][0], image.copy(), word_level_char_bbox.copy(),
        #               region_score.copy(), affinity_score.copy(), confidence_mask.copy())

        image, region_score, affinity_score, confidence_mask = self.augment_image(
            image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        )

        #
        # saveImage(self.img_names[index],"./exp/viz", image.copy(), word_level_char_bbox.copy(),
        #           region_score.copy(), affinity_score.copy(), confidence_mask.copy())
        #


        # saveInput(
        #     self.img_names[index],
        #     image = image,
        #     region_scores = region_score,
        #     affinity_scores = affinity_score,
        #     confidence_mask =confidence_mask
        # )

        # if self.logging:
        #     saveInput(
        #         self.img_names[index][0],
        #         image,
        #         region_score,
        #         affinity_score,
        #         confidence_mask,
        #     )

        # self.logging = False

        region_score = self.resize_to_half(region_score)
        affinity_score = self.resize_to_half(affinity_score)
        confidence_mask = self.resize_to_half(confidence_mask)

        image = imgproc.normalizeMeanVariance(
            np.array(image), mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
        )
        image = image.transpose(2, 0, 1)

        # TODO : region score, affinity score type check
        region_score = region_score.astype(np.float32)
        affinity_score = affinity_score.astype(np.float32)
        confidence_mask = confidence_mask.astype(np.float32)

        return image, region_score, affinity_score, confidence_mask


def test():


    # load configure
    from config.load_config import load_yaml
    from config.load_config import DotDict

    #config = load_yaml('/data/workspace/woans0104/CRAFT-Refactoring/config/syn_test6_26_gnnet')
    config = load_yaml('./syn_new_data')

    config = DotDict(config)

    data_path_kr = "/nas/datahub/SynthText-KR/dset_kr.h5"

    #data_path_kr = "/data/workspace/woans0104/SynthText_kr-master/data/background/gen/syn_kr_v2/dset_kr.h5"

    dataset = SynthTextDataSet_kr(
        output_size=config.train.data.output_size,
        data_dir=data_path_kr,
        saved_gt_dir=config.data_dir.synthtext_gt,
        gauss_init_size=config.train.data.gauss_init_size,
        gauss_sigma=config.train.data.gauss_sigma,
        enlarge_region=config.train.data.enlarge_region,
        enlarge_affinity=config.train.data.enlarge_affinity,
        aug=config.train.data.syn_aug,
        vis_opt=config.train.data.vis_opt,
    )

    import torch
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (image, region_score, affinity_score, confidence_mask) in enumerate(train_loader):
        total += 1
        print(total)
        import ipdb;ipdb.set_trace()