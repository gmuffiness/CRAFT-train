import os
import re
import copy
import random
import numpy as np
import itertools

import cv2
import scipy.io as scio
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

# from utils import config
from utils.util import saveInput, saveImage
from data import imgproc
from data.imgaug import random_scale, random_scale2, random_crop


class SynthTextDataLoader(data.Dataset):
    def __init__(self, args, target_size=768, data_paths="", viz=False):

        self.target_size = target_size
        self.data_paths = data_paths
        self.charbox, self.image, self.imgtxt = self.load_synthtext()
        self.viz = viz

    def load_synthtext(self):

        gt = scio.loadmat(os.path.join(self.data_paths, "gt.mat"))
        wordbox = gt["wordBB"][0]
        charbox = gt["charBB"][0]
        imnames = gt["imnames"][0]
        imgtxt = gt["txt"][0]

        return charbox, imnames, imgtxt

    def load_synthtext_image_gt(self, index):

        # 저장된 region, affinity map을 불러옴
        # 불러온 region map과 동일한 charbox,imnames,imgtxt를 load_synthtext에서 가져옴
        img_path = os.path.join(self.data_paths, self.image[index][0])  # 경로 수정
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        region_score = os.path.join(self.data_paths, self.image[index][0])  # 경로 수정
        affinity_score = os.path.join(self.data_paths, self.image[index][0])  # 경로 수정

        _charbox = copy.deepcopy(self.charbox[index]).transpose((2, 1, 0))
        words = [re.split(" \n|\n |\n| ", t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        rnd_range = [0.5, 1.0, 1.5]
        scale = random.sample(rnd_range, 1)[0]
        image = random_scale2(
            image, min_size=self.target_size, rnd_scale=scale, bboxes=_charbox
        )
        region_score = random_scale2(image, min_size=self.target_size, rnd_scale=scale)
        affinity_score = random_scale2(
            image, min_size=self.target_size, rnd_scale=scale
        )
        confidence_mask = np.ones((image.shape[0], image.shape[1]))

        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total : total + len(words[i])]
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

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def pull_item(self, index):
        (
            image,
            region_score,
            affinity_score,
            character_bboxes,
            words,
            confidence_mask,
            img_path,
        ) = self.load_synthtext_image_gt(index)

        random_transforms = [image, region_score, affinity_score, confidence_mask * 255]
        random_transforms = random_crop(
            random_transforms, (self.target_size, self.target_size), character_bboxes
        )
        image, region_image, affinity_image, confidence_mask = random_transforms

        # resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            saveInput(
                self.image[index][0],
                image,
                region_image,
                affinity_image,
                confidence_mask,
            )
            self.viz = False

        image = Image.fromarray(image)

        if config.AUG == True:
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

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.pull_item(index)
