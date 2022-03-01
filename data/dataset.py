import os
import re
import itertools
import copy

import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import (
    random_crop_with_bbox_adapt_to_output_size,
    random_horizontal_flip,
    random_rotate,
    random_scale,
    random_resize_crop,
)
from data.pointClockOrder import mep
from utils.util import saveInput, saveImage


class SynthTextDataSet(Dataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        gauss_init_size,
        gauss_sigma,
        enlarge_size,
        aug,
        vis_opt,
    ):

        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.img_names, self.char_bbox, self.img_words = self.load_data()
        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_size
        )
        self.aug = aug
        self.vis_opt = vis_opt

    # NOTE
    def load_data(self, bbox="char"):

        gt = scio.loadmat(os.path.join(self.data_dir, "gt.mat"))
        img_names = gt["imnames"][0]
        img_words = gt["txt"][0]

        if bbox == "char":
            img_bbox = gt["charBB"][0]
        else:
            img_bbox = gt["wordBB"][0]  # word bbox needed for test

        return img_names, img_bbox, img_words

    def load_saved_gt(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_char_bbox = self.char_bbox[index].transpose((2, 1, 0))

        image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        region_score_path = os.path.join(
            os.path.join(self.saved_gt_dir, "region/enlarge-1.75"),
            self.img_names[index][0][:-4] + "-region.jpg",
        )
        affinity_score_path = os.path.join(
            os.path.join(self.saved_gt_dir, "affinity/enlarge-1.75"),
            self.img_names[index][0][:-4] + "-affinity.jpg",
        )
        region_score = cv2.imread(region_score_path, cv2.IMREAD_GRAYSCALE)
        affinity_score = cv2.imread(affinity_score_path, cv2.IMREAD_GRAYSCALE)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        words = [
            re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]
        ]
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

    def make_pseudo_gt(self, index):

        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_char_bbox = self.char_bbox[index].transpose((2, 1, 0))
        image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        img_h, img_w, _ = image.shape

        confidence_mask = np.ones((img_h, img_w), dtype=np.uint8)

        words = [
            re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]
        ]
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

        region_score = self.gaussian_builder.generate_region(
            img_h, img_w, word_level_char_bbox
        )
        affinity_score, _ = self.gaussian_builder.generate_affinity(
            img_h, img_w, word_level_char_bbox
        )

        # TODO: output validation check

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
                == "random_crop_with_bbox_adapt_to_output_size"
            ):
                augment_targets = random_crop_with_bbox_adapt_to_output_size(
                    augment_targets, word_level_char_bbox, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop":
                augment_targets = random_resize_crop(
                    augment_targets,
                    self.aug.random_crop.scale,
                    self.aug.random_crop.ratio,
                    self.output_size,
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

        if self.saved_gt_dir is None:
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

        # if self.logging:
        #     saveImage(self.img_names[index][0], image.copy(), word_level_char_bbox.copy(),
        #               region_score.copy(), affinity_score.copy(), confidence_mask.copy())

        image, region_score, affinity_score, confidence_mask = self.augment_image(
            image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        )

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
        region_score = region_score.astype(np.float32) / 255
        affinity_score = affinity_score.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32)

        return image, region_score, affinity_score, confidence_mask


class ICDAR2015(Dataset):
    def __init__(self, output_size, data_dir, saved_gt_dir, gauss_init_size, gauss_sigma, enlarge_size, aug,
                 vis_opt):

        # self.net = net
        # self.net.eval()
        # self.net = 0

        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.gaussian_builder = GaussianBuilder(gauss_init_size, gauss_sigma, enlarge_size)
        self.aug = aug
        self.vis_opt = vis_opt
        self.vis_index = [189, 41, 723, 251, 232, 115, 634, 951, 247, 25, 400, 704, 619, 305, 423, 20, 31]

        self.img_dir = os.path.join(data_dir, 'ch4_training_images')
        self.img_gt_box_dir = os.path.join(data_dir, 'ch4_training_localization_transcription_gt')
        self.image_names = os.listdir(self.img_dir)

    def get_img_name(self, index):
        return self.image_names[index]

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding='utf-8').readlines()
        word_bboxes = []
        words = []
        for line in lines:
            box_info = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            # int type
            box_points = [int(box_info[i]) for i in range(8)]
            word = box_info[8:]
            word = ','.join(word)
            # np.int32 type
            box_points = np.array(box_points, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                word_bboxes.append(box_points)
                continue

            # TODO: 좌표 보정 과정으로 보이는데, 어떤 점이 달라지는 지 확인
            area, p0, p3, p2, p1, _, _ = mep(box_points)

            bbox = np.array([p0, p1, p2, p3])

            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box_points[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            new_box = np.array(new_box)
            word_bboxes.append(np.array(new_box))
            words.append(word)
        return word_bboxes, words

    def crop_image_by_bbox(self, image, box):

        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(
                                                np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(
                                                np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M


    def load_saved_gt(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0])
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)
        word_bboxes = np.float32(word_bboxes)

        query_idx = int(self.get_img_name(index).split('.')[0].split('_')[1])
        # use official CRAFT model's output as teacher (to make pseudo-label)
        saved_region_scores_path = os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_region.jpg')
        saved_affi_scores_path = os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_affi.jpg')
        region_score = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        affinity_score = cv2.imread(saved_affi_scores_path, cv2.IMREAD_GRAYSCALE)

        region_score = cv2.resize(region_score, (image.shape[1], image.shape[0])).astype(np.float32)
        affinity_score = cv2.resize(affinity_score, (image.shape[1], image.shape[0])).astype(np.float32)

        saved_cf_mask_path = os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_cf_mask_thresh_0.6.jpg')
        confidence_mask = cv2.imread(saved_cf_mask_path, cv2.IMREAD_GRAYSCALE)
        confidence_mask = cv2.resize(confidence_mask, (image.shape[1], image.shape[0])).astype(np.float32)

        # 기존 code 중 아래 random_crop에서 쓰이게 될 character bboxes 형식을 맞춰주기 위해,
        # word bboxes를 1개의 character씩 담긴 bboxes로 만들어 줌

        word_level_char_bbox = []
        trunc_mask = np.zeros([image.shape[0], image.shape[1]])
        for i in range(len(word_bboxes)):
            cv2.fillPoly(trunc_mask, [np.int32(word_bboxes[i])], 1)
            if (word_bboxes[i] < 0).sum() > 0:
                word_bboxes[i] = np.where(word_bboxes[i] < 0, 0, word_bboxes[i])
            word_level_char_bbox.append(np.expand_dims(word_bboxes[i], 0))

        # truncate region, affinity out of GT box
        trunc_mask = trunc_mask.astype(np.float32)
        region_score = region_score * trunc_mask
        affinity_score = affinity_score * trunc_mask

        # check minus coordinate
        for cb in word_level_char_bbox:
            if (cb < 0).astype('float32').sum() > 0:
                import ipdb;
                ipdb.set_trace()
                print(query_idx)


        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        )

    def augment_image(self, image, region_score, affinity_score, confidence_mask, word_level_char_bbox):

        augment_targets = [image, region_score, affinity_score, confidence_mask]

        if self.aug.random_scale.option:
            augment_targets, word_level_char_bbox = random_scale(augment_targets, word_level_char_bbox,
                                                                 self.aug.random_scale.range)

        if self.aug.random_rotate.option:
            augment_targets = random_rotate(augment_targets, self.aug.random_rotate.max_angle)

        if self.aug.random_crop.option:
            if self.aug.random_crop.version == "random_crop_with_bbox_adapt_to_output_size":
                augment_targets = random_crop_with_bbox_adapt_to_output_size(
                    augment_targets, word_level_char_bbox, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop":
                augment_targets = random_resize_crop(
                    augment_targets, self.aug.random_crop.scale, self.aug.random_crop.ratio, self.output_size
                )
            else:
                assert "Undefined RandomCrop version"

        if self.aug.random_horizontal_flip.option:
            augment_targets = random_horizontal_flip(augment_targets)

        if self.aug.random_colorjitter.option:
            image, region_score, affinity_score, confidence_mask = augment_targets
            image = Image.fromarray(image)
            image = transforms.ColorJitter(brightness=self.aug.random_colorjitter.brightness,
                                           contrast=self.aug.random_colorjitter.contrast,
                                           saturation=self.aug.random_colorjitter.saturation,
                                           hue=self.aug.random_colorjitter.hue)(image)
        else:
            image, region_score, affinity_score, confidence_mask = augment_targets

        return np.array(image), region_score, affinity_score, confidence_mask

    def resize_to_half(self, ground_truth):
        return cv2.resize(ground_truth, (self.output_size // 2, self.output_size // 2))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        if self.saved_gt_dir is None:
            pass
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
        confidence_mask = confidence_mask.astype(np.float32)

        return image, region_score, affinity_score, confidence_mask
