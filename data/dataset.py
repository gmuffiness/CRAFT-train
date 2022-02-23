import os
import re
import itertools

import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data import imgproc
from data.imgaug import random_crop_with_bbox_adapt_to_output_size
from utils.util import saveInput, saveImage


class SynthTextDataSet(Dataset):
    def __init__(self, output_size, data_dir, saved_gt_dir, logging):

        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.img_names, self.char_bbox, self.img_words = self.load_data()
        self.logging = logging

    # NOTE
    def load_data(self, bbox="char"):

        gt = scio.loadmat(os.path.join(self.data_dir, "gt.mat"))
        img_names = gt["imnames"][0]
        img_words = gt["txt"][0]

        if bbox == "char" :
            img_bbox = gt["charBB"][0]
        else: img_bbox = gt["wordBB"][0] # word bbox needed for test


        return img_names, img_bbox, img_words

    def load_saved_gt(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_char_bbox = self.char_bbox[index].transpose((2, 1, 0))

        image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        # EDIT when saved scores are ready
        # region_score = os.path.join(self.saved_gt_dir, self.img_names[index][0])
        # affinity_score = os.path.join(self.saved_gt_dir, self.img_names[index][0])
        region_score = image[:,:,0]
        affinity_score = image[:,:,0]

        confidence_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        words = [re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]]
        words = list(itertools.chain(*words))
        words = [word for word in words if len(word) > 0]

        word_level_char_bbox = []
        char_idx = 0
        for i in range(len(words)):
            length_of_word = len(words[i])
            word_bbox = all_char_bbox[char_idx : char_idx + length_of_word]
            assert len(word_bbox) == length_of_word
            char_idx += length_of_word
            word_bbox = np.array(word_bbox)
            word_level_char_bbox.append(word_bbox)

        # TODO: output validation check

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        )

    # TODO
    def make_pseudo_gt(self, index):
        return 0

    def dilate_img_to_output_size(self, image, char_bbox):
        h, w = image.shape[0:2]
        if min(h, w) <= self.output_size:
            scale = float(self.output_size + 10) / min(h, w)
        else:
            scale = 1.0
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        char_bbox *= scale
        return image, char_bbox

    def augment_image(self, image, region_score, affinity_score, confidence_mask, word_level_char_bbox):

        augment_targets = [image, region_score, affinity_score, confidence_mask]

        # TODO
        # 1. rotate

        # 2. scale

        # 3. crop
        augment_targets = random_crop_with_bbox_adapt_to_output_size(
            augment_targets, word_level_char_bbox, self.output_size
        )

        # 4. horizontal flip

        # 5. colorjitter
        image, region_image, affinity_image, confidence_mask = augment_targets

        image = Image.fromarray(image)
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        return image, region_score, affinity_score, confidence_mask

    def resize_to_half(self, ground_truth):
        return cv2.resize(ground_truth, (self.output_size // 2, self.output_size // 2))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        if self.saved_gt_dir == "":
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                words,
            ) = self.make_pseudo_gt(index)
        else:
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                words,
            ) = self.load_saved_gt(index)

            image, region_score, affinity_score, confidence_mask = \
                self.augment_image(image, region_score, affinity_score, confidence_mask, word_level_char_bbox)


        # NOTE
        # if cfg.train.data.aug:
        #     image, region_score, affinity_score, confidence_mask = \
        #         self.augment_image(image, region_score, affinity_score, confidence_mask, word_level_char_bbox)

        if self.logging:
            saveInput(
                self.img_names[index][0],
                image,
                region_score,
                affinity_score,
                confidence_mask,
            )
            self.logging = False

        region_score = self.resize_to_half(region_score)
        affinity_score = self.resize_to_half(affinity_score)
        confidence_mask = self.resize_to_half(confidence_mask)

        image = imgproc.normalizeMeanVariance(
            np.array(image), mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
        )
        image = image.transpose(2, 0, 1)

        # TODO : region score, affinity score type check
        region_score = region_score.astype(np.float32) / 255
        affinity_score = affinity_score.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) / 255

        return image, region_score, affinity_score, confidence_mask