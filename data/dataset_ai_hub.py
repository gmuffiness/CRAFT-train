import os
import json

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
    random_resize_crop_ai
)
from utils.util import saveInput, saveImage
from data.pseudo_label.make_charbox import PseudoCharBoxBuilder


from functools import wraps
import time




class AiHubDataset(Dataset):
    def __init__(
        self,
        output_size,
        data_dir,
        gt_path,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_opt,
    ):

        self.output_size = output_size
        self.data_dir = data_dir

        self.gt_data = self.load_data(gt_path)


        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )
        self.aug = aug
        self.vis_opt = vis_opt


    def load_data(self, gt_path):

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        return gt_data


    def make_char_bbox(self,bbox):

        char_bbox = np.ndarray((4, 2), np.int)
        box = np.array(bbox)
        char_x = box[0]
        char_y = box[1]
        char_width = box[2]
        char_height = box[3]

        char_bbox[0][0] = char_x
        char_bbox[0][1] = char_y

        # upper-right corner
        char_bbox[1][0] = char_x + char_width
        char_bbox[1][1] = char_y

        # lower-right corner
        char_bbox[2][0] = char_x + char_width
        char_bbox[2][1] = char_y + char_height

        # lower-left corner
        char_bbox[3][0] = char_x
        char_bbox[3][1] = char_y + char_height

        return char_bbox



    def load_img_gt_box(self, index):

        word_bboxes = []
        words = []

        do_not_care_words = []
        do_not_care_bboxes = []
        vertical_word = []

        for j in range(len(self.gt_data[index][0]["annotation"])):
            if self.gt_data[index][0]["annotation"][j]["text"] != "###":
                words.append(self.gt_data[index][0]["annotation"][j]["text"])
                vertical_word.append(self.gt_data[index][0]["annotation"][j]["vertical"])
                char_bbox_per_words = []
                for k in range(len(self.gt_data[index][0]["annotation"][j]["bbox"])):
                    word_bbox = self.make_char_bbox(self.gt_data[index][0]["annotation"][j]["bbox"][k])
                    char_bbox_per_words.append(word_bbox)
                word_bboxes.append(char_bbox_per_words)

            else:
                do_not_care_words.append(self.gt_data[index][0]["annotation"][j]["text"])
                for k in range(len(self.gt_data[index][0]["annotation"][j]["bbox"])):
                    do_not_care_bbox = self.make_char_bbox(self.gt_data[index][0]["annotation"][j]["bbox"][k])
                    do_not_care_bboxes.append(do_not_care_bbox)


        return word_bboxes, words, do_not_care_bboxes, do_not_care_words, vertical_word




    def make_pseudo_gt(self, index):


        img_id = self.gt_data[index][0]["image_id"]
        img_path = os.path.join(self.data_dir, self.gt_data[index][0]["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox, do_care_words, do_not_care_bboxes, do_not_care_words, vertical_word \
            = self.load_img_gt_box(index)


        if len(word_level_char_bbox) == 0 and len(do_not_care_bboxes)==0:
            import ipdb;ipdb.set_trace()
            print(img_id)
            return image, word_level_char_bbox, do_care_words, confidence_mask

        for i in range(len(do_not_care_bboxes)):
            if do_not_care_words[i] == "###" or len(do_not_care_words[i].strip()) == 0:
                cv2.fillPoly(confidence_mask, [np.int32(do_not_care_bboxes[i])], 0)
                continue


        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
        else:
            region_score = self.gaussian_builder.generate_region(
                img_h, img_w, word_level_char_bbox,
                horizontal_text_bools=[True for _ in range(len(do_care_words))]
            )
            affinity_score, _ = self.gaussian_builder.generate_affinity_ai(
                img_h, img_w, word_level_char_bbox, vertical=vertical_word,
                horizontal_text_bools=[True for _ in range(len(do_care_words))]

            )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            do_care_words
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

        return len(self.gt_data)

    def __getitem__(self, index):

        (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words
        ) = self.make_pseudo_gt(index)



        # if self.logging:
        #     saveImage(self.img_names[index][0], image.copy(), word_level_char_bbox.copy(),
        #               region_score.copy(), affinity_score.copy(), confidence_mask.copy())

        image, region_score, affinity_score, confidence_mask = self.augment_image(
            image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        )

        # save_img_name = self.gt_data[index][0]["file_name"]
        #
        # saveImage(save_img_name,"./exp/viz/ai-test", image.copy(), word_level_char_bbox.copy(),
        #           region_score.copy(), affinity_score.copy(), confidence_mask.copy())


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
    import torch

    config = load_yaml('./syn_new_data')


    config = DotDict(config)


    dataloader = AiHubDataset(
        output_size=config.train.data.output_size,
        data_dir='/nas/datahub/ai-hub-data/textinthewild_data/all_image',
        gt_path='/nas/datahub/ai-hub-data/textinthewild_data/last_new_a100.json',
        gauss_init_size=config.train.data.gauss_init_size,
        gauss_sigma=config.train.data.gauss_sigma,
        enlarge_region=config.train.data.enlarge_region,
        enlarge_affinity=config.train.data.enlarge_affinity,
        aug=config.train.data.syn_aug,
        vis_opt=config.train.data.vis_opt)


    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=48,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (image, region_score, affinity_score, confidence_mask) in enumerate(train_loader):
        total += 1
        print(total)
        import ipdb;ipdb.set_trace()

        if total % 100 == 0:
            import ipdb;ipdb.set_trace()
