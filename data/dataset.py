import os
import re
import copy
import random
import numpy as np
import itertools
import time

import cv2
import scipy.io as scio
from PIL import Image
import torch
import torchvision.transforms as transforms

from config.load_config import cfg
from utils.util import saveInput, saveImage
from data import imgproc
from data.imgaug import random_scale, random_scale2, random_crop


class SynthTextDataLoader(torch.utils.data.Dataset):
    def __init__(self, output_size, data_dir, saved_gt_dir, logging):

        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.img_names, self.char_bbox, self.img_words = self.load_data()
        self.logging = logging

    def load_data(self):
        gt = scio.loadmat(os.path.join(self.data_dir, "gt.mat"))
        img_names = gt["imnames"][0]
        char_bbox = gt["charBB"][0]
        img_words = gt["txt"][0]
        return img_names, char_bbox, img_words

    def load_saved_gt(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        char_bbox = self.char_bbox[index].transpose((2, 1, 0))

        image, char_bbox = self.dilate_img_to_output_size(image, char_bbox)

        # region_score = os.path.join(self.saved_gt_dir, self.img_names[index][0])
        # affinity_score = os.path.join(self.saved_gt_dir, self.img_names[index][0])
        region_score = image
        affinity_score = image

        words = [re.split(" \n|\n |\n| ", t.strip()) for t in self.img_words[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        import ipdb; ipdb.set_trace()

        confidence_mask = np.ones((image.shape[0], image.shape[1]))

        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = char_bbox[total : total + len(words[i])]
            assert len(bboxes) == len(words[i])
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return (
            image,
            region_score,
            affinity_score,
            character_bboxes,
            words,
            confidence_mask,
            img_path,
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


    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.output_size // 2, self.output_size // 2))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        if self.saved_gt_dir == "":
            (
                image,
                region_score,
                affinity_score,
                character_bboxes,
                words,
                confidence_mask,
                img_path,
            ) = self.make_pseudo_gt(index)
        else:
            (
                image,
                region_score,
                affinity_score,
                character_bboxes,
                words,
                confidence_mask,
                img_path,
            ) = self.load_saved_gt(index)

        random_transforms = [image, region_score, affinity_score, confidence_mask * 255]
        random_transforms = random_crop(
            random_transforms, (self.output_size, self.output_size), character_bboxes
        )
        image, region_image, affinity_image, confidence_mask = random_transforms

        # resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.logging:
            saveInput(
                self.img_names[index][0],
                image,
                region_image,
                affinity_image,
                confidence_mask,
            )
            self.logging = False

        image = Image.fromarray(image)

        if cfg.train.data.aug:
            image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
            # image = transforms.ColorJitter(brightness=32.0 / 255, contrast=0.5, saturation=0.5, hue=0.25)(image)
        image = imgproc.normalizeMeanVariance(
            np.array(image), mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
        )
        image = image.transpose(2, 0, 1)

        region_image = region_image.astype(np.float32) / 255
        affinity_image = affinity_image.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) / 255

        return image, region_image, affinity_image, confidence_mask
