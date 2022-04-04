import os
import re
import itertools
import random
import json
import math

import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py

from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import (
    rescale,
    random_resize_crop_synth,
    random_resize_crop,
    random_crop_with_bbox,
    random_horizontal_flip,
    random_rotate,
    random_scale,
    random_crop
)
from data.pseudo_label.make_charbox import PseudoCharBoxBuilder
from utils.util import saveInput, saveImage
from data.boxEnlarge import enlargebox
from utils.decorator_wraps import time_printer
from shapely.geometry import Polygon
from shapely.geometry import box
from torchvision.transforms import RandomResizedCrop, RandomCrop


def hierarchical_dataset(root, config, select_data="/"):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []

    print(f"dataset_root:    {root}\t dataset: {select_data[0]}")
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        for i in filenames:
            lmdb_path = os.path.join(dirpath, str(i))
            dataset = SynthTextDataSet_KR(
                output_size=config.train.data.output_size,
                data_dir=lmdb_path,
                saved_gt_dir=None,
                mean=config.train.data.mean,
                variance=config.train.data.variance,
                gauss_init_size=config.train.data.gauss_init_size,
                gauss_sigma=config.train.data.gauss_sigma,
                enlarge_region=config.train.data.enlarge_region,
                enlarge_affinity=config.train.data.enlarge_affinity,
                aug=config.train.data.syn_kor_aug,
                vis_test_dir=config.vis_test_dir,
                vis_opt=config.train.data.vis_opt,
                sample=config.train.data.syn_sample,
            )

            print(
                f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
            )
            dataset_list.append(dataset)

    return dataset_list


class CraftBaseDataset(Dataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
    ):
        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.mean, self.variance = mean, variance
        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )
        self.aug = aug
        self.vis_test_dir = vis_test_dir
        self.vis_opt = vis_opt
        self.sample = sample
        if self.sample != -1:
            random.seed(0)
            self.idx = random.sample(range(0, len(self.img_names)), self.sample)

        self.pre_crop_area = []

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
            if self.aug.random_crop.version == "random_crop_with_bbox":
                augment_targets = random_crop_with_bbox(
                    augment_targets, word_level_char_bbox, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop_synth":
                augment_targets = random_resize_crop_synth(
                    augment_targets, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop":

                if len(self.pre_crop_area) > 0 :
                    pre_crop_area = self.pre_crop_area
                else:
                    pre_crop_area = None

                augment_targets = random_resize_crop(
                    augment_targets,
                    self.aug.random_crop.scale,
                    self.aug.random_crop.ratio,
                    self.output_size,
                    self.aug.random_crop.rnd_threshold,
                    pre_crop_area
                )


            elif self.aug.random_crop.version == "random_crop":
                augment_targets = random_crop(
                    augment_targets,
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

    def resize_to_half(self, ground_truth, interpolation):
        return cv2.resize(
            ground_truth,
            (self.output_size // 2, self.output_size // 2),
            interpolation=interpolation,
        )

    def __len__(self):
        if self.sample != -1:
            return len(self.idx)
        else:
            return len(self.img_names)

    def __getitem__(self, index):
        # index = self.img_names.index(f'img_{self.vis_index[self.temp_idx]}.jpg')
        # self.temp_idx += 1
        # index = self.img_names.index('img_274.jpg')
        if self.sample != -1:
            index = self.idx[index]
        if self.saved_gt_dir is None:
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                all_affinity_bbox,
                words,
            ) = self.make_gt_score(index)
        else:
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                words,
            ) = self.load_saved_gt_score(index)
            all_affinity_bbox = []

        # To visualize only predefined index of ICDAR15 dataset
        # query_idx = int(self.img_names[index].split(".")[0].split("_")[1])
        # if self.vis_opt and query_idx in self.vis_index:

        if self.vis_opt:
            saveImage(
                self.img_names[index],
                self.vis_test_dir,
                image.copy(),
                word_level_char_bbox.copy(),
                all_affinity_bbox.copy(),
                region_score.copy(),
                affinity_score.copy(),
                confidence_mask.copy(),
            )

        image, region_score, affinity_score, confidence_mask = self.augment_image(
            image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        )

        if self.vis_opt:
            saveInput(
                self.img_names[index],
                self.vis_test_dir,
                image,
                region_score,
                affinity_score,
                confidence_mask,
            )

        region_score = self.resize_to_half(region_score, interpolation=cv2.INTER_CUBIC)
        affinity_score = self.resize_to_half(
            affinity_score, interpolation=cv2.INTER_CUBIC
        )
        confidence_mask = self.resize_to_half(
            confidence_mask, interpolation=cv2.INTER_NEAREST
        )

        image = imgproc.normalizeMeanVariance(
            np.array(image), mean=self.mean, variance=self.variance
        )
        image = image.transpose(2, 0, 1)

        return image, region_score, affinity_score, confidence_mask


class SynthTextDataSet(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.img_names, self.char_bbox, self.img_words = self.load_data()
        self.vis_index = list(range(1000))

    # TODO: load data with generator will save more train preparing time?
    def load_data(self, bbox="char"):

        gt = scio.loadmat(os.path.join(self.data_dir, "gt.mat"))
        img_names = gt["imnames"][0]
        img_words = gt["txt"][0]

        if bbox == "char":
            img_bbox = gt["charBB"][0]
        else:
            img_bbox = gt["wordBB"][0]  # word bbox needed for test

        return img_names, img_bbox, img_words

    def dilate_img_to_output_size(self, image, char_bbox):
        h, w, _ = image.shape
        if min(h, w) <= self.output_size:
            scale = float(self.output_size) / min(h, w)
        else:
            scale = 1.0
        image = cv2.resize(
            image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        char_bbox *= scale
        return image, char_bbox

    def make_gt_score(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_char_bbox = self.char_bbox[index].transpose(
            (2, 1, 0)
        )  # shape : (Number of characters in image, 4, 2)
        # image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        img_h, img_w, _ = image.shape

        confidence_mask = np.ones((img_h, img_w), dtype=np.float32)

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
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )
        affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )


class SynthTextDataSet_KR(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.gt = None
        with h5py.File(self.data_dir, "r") as file:
            self.img_names = np.array(list(file["data"].keys()))

    def load_data(self, path):
        folder, ext = os.path.splitext(path)
        if ext == ".h5":
            gt = h5py.File(path, "r")
        else:
            gt = h5py.File(os.path.join(path, "dset_kr.h5"), "r")

        return gt

    @property
    def get_gt(self):
        if self.gt is None:
            self.gt = self.load_data(self.data_dir)
        return self.gt

    def dilate_img_to_output_size(self, image, char_bbox):
        h, w, _ = image.shape
        if min(h, w) <= self.output_size:
            scale = float(self.output_size) / min(h, w)
        else:
            scale = 1.0
        image = cv2.resize(
            image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        char_bbox *= scale
        return image, char_bbox

    def make_gt_score(self, index):
        gt = self.get_gt["data"][self.img_names[index]]

        image = gt[...]  # RGB

        # print('img size : {}'.format(image.shape))

        charBB = gt.attrs["charBB"]
        txt = gt.attrs["txt"]

        all_char_bbox = charBB.transpose((2, 1, 0))
        # image, all_char_bbox = self.dilate_img_to_output_size(image, all_char_bbox)

        img_h, img_w, _ = image.shape
        confidence_mask = np.ones((img_h, img_w), np.float32)

        try:
            words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]
        except:
            txt = [t.decode("UTF-8") for t in txt]
            words = [re.split(" \n|\n |\n| ", t.strip()) for t in txt]

        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

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
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )
        affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )


class AiHubDataset(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
        do_not_care_label,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.do_not_care_label = do_not_care_label
        self.img_dir = os.path.join(data_dir, "all_image")
        self.img_gt_box_json_path = os.path.join(
            data_dir, "last_new_a100.json"
        )
        self.img_gt_box = self.load_data(self.img_gt_box_json_path)
        self.img_names = []
        for i in range(len(self.img_gt_box)):
            self.img_names.append(self.img_gt_box[i][0]['file_name'])

    def load_data(self, gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        return gt_data

    def make_char_bbox(self, bbox):
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

    def cal_angle(self, v1):
        theta = np.arccos(min(1, v1[0] / (np.linalg.norm(v1) + 10e-8)))
        return 2 * math.pi - theta if v1[1] < 0 else theta

    def clockwise_sort(self, points):
        # return 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
        v1, v2, v3, v4 = points
        center = (v1 + v2 + v3 + v4) / 4
        theta = np.array([self.cal_angle(v1 - center), self.cal_angle(v2 - center), \
                          self.cal_angle(v3 - center), self.cal_angle(v4 - center)])
        index = np.argsort(theta)
        return np.array([v1, v2, v3, v4])[index, :]

    def load_img_gt_box(self, index):

        word_bboxes = []
        words = []

        do_not_care_words = []
        do_not_care_bboxes = []
        vertical_word = []

        for j in range(len(self.img_gt_box[index][0]["annotation"])):
            if self.img_gt_box[index][0]["annotation"][j]["text"] not in self.do_not_care_label:
                words.append(self.img_gt_box[index][0]["annotation"][j]["text"])
                vertical_word.append(
                    self.img_gt_box[index][0]["annotation"][j]["vertical"]
                )
                char_bbox_per_words = []
                for k in range(len(self.img_gt_box[index][0]["annotation"][j]["bbox"])):
                    word_bbox = self.make_char_bbox(
                        self.img_gt_box[index][0]["annotation"][j]["bbox"][k]
                    )
                    char_bbox_per_words.append(word_bbox)
                word_bboxes.append(char_bbox_per_words)

            else:
                do_not_care_words.append(
                    self.img_gt_box[index][0]["annotation"][j]["text"]
                )
                for k in range(len(self.img_gt_box[index][0]["annotation"][j]["bbox"])):
                    do_not_care_bbox = self.make_char_bbox(
                        self.img_gt_box[index][0]["annotation"][j]["bbox"][k]
                    )
                    do_not_care_bboxes.append(do_not_care_bbox)

        return word_bboxes, words, do_not_care_bboxes, do_not_care_words, vertical_word

    def make_gt_score(self, index):
        img_id = self.img_gt_box[index][0]["image_id"]
        img_path = os.path.join(self.img_dir, self.img_gt_box[index][0]["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        (
            word_level_char_bbox,
            do_care_words,
            do_not_care_bboxes,
            do_not_care_words,
            vertical_word,
        ) = self.load_img_gt_box(index)
        for i in range(len(do_not_care_bboxes)):
            cv2.fillPoly(confidence_mask, [np.int32(do_not_care_bboxes[i])], 0)

        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
            all_affinity_bbox = []
        else:
            horizontal_text_bools = [not bool_val for bool_val in vertical_word]
            region_score = self.gaussian_builder.generate_region(
                img_h,
                img_w,
                word_level_char_bbox,
                horizontal_text_bools=[True for _ in range(len(do_care_words))],
            )
            (
                affinity_score,
                all_affinity_bbox,
            ) = self.gaussian_builder.generate_affinity_ai(
                img_h,
                img_w,
                word_level_char_bbox,
                vertical=vertical_word,
                horizontal_text_bools=horizontal_text_bools,
            )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            do_care_words,
        )


class ICDAR2015(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
        watershed_param,
        pseudo_vis_opt,
        do_not_care_label,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.pseudo_vis_opt = pseudo_vis_opt
        self.do_not_care_label = do_not_care_label
        self.pseudo_charbox_builder = PseudoCharBoxBuilder(
            watershed_param, vis_test_dir, pseudo_vis_opt, self.gaussian_builder
        )
        # self.vis_index = [189, 41, 723, 251, 232, 115, 634, 951, 247, 25, 400, 704, 619, 305, 423, 20, 31, 61, 73]
        # self.vis_index = [61, 73, 77, 88, 94, 126,131,136,163,208,245,274,283,297,337,354,356,359,371,372,378,410,435,443,450]
        self.vis_index = list(range(1000))
        self.img_dir = os.path.join(data_dir, "ch4_training_images")
        self.img_gt_box_dir = os.path.join(
            data_dir, "ch4_training_localization_transcription_gt"
        )
        self.img_names = os.listdir(self.img_dir)

    def update_model(self, net):
        self.net = net

    def update_device(self, gpu):
        self.gpu = gpu

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding="utf-8").readlines()
        word_bboxes = []
        words = []
        for line in lines:
            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box_points = [int(box_info[i]) for i in range(8)]
            box_points = np.array(box_points, np.float32).reshape(4, 2)
            word = box_info[8:]
            word = ",".join(word)
            if word in self.do_not_care_label:
                words.append(self.do_not_care_label[0])
                word_bboxes.append(box_points)
                continue
            word_bboxes.append(box_points)
            words.append(word)
        return np.array(word_bboxes), words

    def load_data(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(
            img_gt_box_path
        )  # shape : (Number of word bbox, 4, 2)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox = []
        do_care_words = []
        horizontal_text_bools = []

        if len(word_bboxes) == 0:
            return (
                image,
                word_level_char_bbox,
                do_care_words,
                confidence_mask,
                horizontal_text_bools,
            )
        _word_bboxes = word_bboxes.copy()
        for i in range(len(word_bboxes)):
            # _word_bboxes[i] = enlargebox(_word_bboxes[i], image.shape[0], image.shape[1], [0.5, 0.5])
            # TODO: fill confidence mask 할 때, 더 낮은 값이 들어가도록 수정?
            if words[i] in self.do_not_care_label:
                cv2.fillPoly(confidence_mask, [np.int32(_word_bboxes[i])], 0)
                continue

            (
                pseudo_char_bbox,
                confidence,
                horizontal_text_bool,
            ) = self.pseudo_charbox_builder.build_char_box(
                self.net, self.gpu, image, word_bboxes[i], words[i], img_name=img_name
            )

            cv2.fillPoly(confidence_mask, [np.int32(_word_bboxes[i])], confidence)
            do_care_words.append(words[i])
            word_level_char_bbox.append(pseudo_char_bbox)
            horizontal_text_bools.append(horizontal_text_bool)

        return (
            image,
            word_level_char_bbox,
            do_care_words,
            confidence_mask,
            horizontal_text_bools,
        )

    def make_gt_score(self, index):
        """
        Make region, affinity scores using pseudo character-level GT bounding box
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """
        (
            image,
            word_level_char_bbox,
            words,
            confidence_mask,
            horizontal_text_bools,
        ) = self.load_data(index)
        img_h, img_w, _ = image.shape

        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
            all_affinity_bbox = []
        else:
            region_score = self.gaussian_builder.generate_region(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )
            affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )

    def load_saved_gt_score(self, index):
        """
        Load pre-saved official CRAFT model's region, affinity scores to train IC15
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)
        image, word_bboxes = rescale(image, word_bboxes)
        img_h, img_w, _ = image.shape

        query_idx = int(self.img_names[index].split(".")[0].split("_")[1])

        saved_region_scores_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_region.jpg"
        )
        saved_affi_scores_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_affi.jpg"
        )
        saved_cf_mask_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_cf_mask_thresh_0.6.jpg"
        )
        region_score = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        affinity_score = cv2.imread(saved_affi_scores_path, cv2.IMREAD_GRAYSCALE)
        confidence_mask = cv2.imread(saved_cf_mask_path, cv2.IMREAD_GRAYSCALE)

        region_score = cv2.resize(region_score, (img_w, img_h))
        affinity_score = cv2.resize(affinity_score, (img_w, img_h))
        confidence_mask = cv2.resize(
            confidence_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

        region_score = region_score.astype(np.float32) / 255
        affinity_score = affinity_score.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) / 255

        # NOTE : Even though word_level_char_bbox is not necessary, align bbox format with make_gt_score()
        word_level_char_bbox = []

        for i in range(len(word_bboxes)):
            word_level_char_bbox.append(np.expand_dims(word_bboxes[i], 0))

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        )



class PreScripTion(CraftBaseDataset):
    def __init__(
            self,
            output_size,
            data_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            watershed_param,
            pseudo_vis_opt,
            saved_gt_dir=None,
            sample=-1


    ):

        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )

        self.output_size = output_size
        self.data_dir = data_dir
        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )
        self.pseudo_charbox_builder = PseudoCharBoxBuilder(
            watershed_param, vis_test_dir, pseudo_vis_opt, self.gaussian_builder
        )
        self.aug = aug
        self.vis_test_dir = vis_test_dir
        self.vis_opt = vis_opt
        self.pseudo_vis_opt = pseudo_vis_opt
        self.vis_index = [189, 41, 723, 251, 232, 115, 634, 951, 247, 25, 400, 704, 619, 305, 423, 20, 31]

        self.img_dir = data_dir
        self.img_names = [i for i in os.listdir(self.img_dir) if i.endswith(".jpg")]



    def update_model(self, net):
        self.net = net

    def update_device(self, gpu):
        self.gpu = gpu

    def check_label(self, box):

        check = True
        w = max(
            int(np.linalg.norm(box[0] - box[1])), int(np.linalg.norm(box[2] - box[3]))
        )
        h = max(
            int(np.linalg.norm(box[0] - box[3])), int(np.linalg.norm(box[1] - box[2]))
        )
        try:
            word_ratio = h / w
        except:
            check = False

        return check

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding="utf-8").readlines()
        word_bboxes = []
        words = []
        for line in lines:

            box_info, word = line.strip().encode("utf-8").decode("utf-8-sig").split("##::")
            box_info = box_info.strip().encode("utf-8").decode("utf-8-sig").split(" ")
            box_points = [int(box_info[i]) for i in range(8)]
            box_points = np.array(box_points, np.float32).reshape(4, 2)

            if word == "dnc":
                words.append("###")
                word_bboxes.append(box_points)
                continue
            word_bboxes.append(box_points)
            words.append(word)
        return np.array(word_bboxes), words

    def load_data(self, index):

        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_dir, "%s_label.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)  # shape : (Number of word bbox, 4, 2)

        # rescale
        image, word_bboxes = rescale(image, word_bboxes, target_size=2560)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox = []
        do_care_words = []
        horizontal_text_bools = []

        if len(word_bboxes) == 0:
            return image, word_level_char_bbox, do_care_words, confidence_mask


        #------------------------------------------------------------------------------------#


        pre_crop_top, pre_crop_left, pre_crop_width,pre_crop_height  \
            = RandomResizedCrop.get_params(Image.fromarray(image), scale=self.aug.random_crop.scale,
                                                     ratio=self.aug.random_crop.ratio)

        self.pre_crop_area = []
        self.pre_crop_area.extend([pre_crop_top,pre_crop_left,pre_crop_width,pre_crop_height])


        # shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)
        pre_crop_area = box(pre_crop_left, pre_crop_top,
                            pre_crop_left+pre_crop_width, pre_crop_top+pre_crop_height, ccw=False)

        # ------------------------------------------------------------------------------------#

        for i in range(len(word_bboxes)):
            # TODO: fill confidence mask 할 때, 더 낮은 값이 들어가도록 수정?
            if words[i] == "###" or len(words[i].strip()) == 0:
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], 0)
                continue

        # ------------------------------------------------------------------------------#

            word_poly = Polygon(word_bboxes[i])

            if word_poly.area == 0 or pre_crop_area.intersects(word_poly) == False :
                continue

            # ------------------------------------------------------------------------------#
            pseudo_char_bbox, confidence, horizontal_text_bool = self.pseudo_charbox_builder.build_char_box(
                self.net, self.gpu, image, word_bboxes[i], words[i], img_name=img_name
            )

            cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], confidence)
            do_care_words.append(words[i])
            word_level_char_bbox.append(pseudo_char_bbox)
            horizontal_text_bools.append(horizontal_text_bool)

        return image, word_level_char_bbox, do_care_words, confidence_mask, horizontal_text_bools

    def make_gt_score(self, index):
        """
        Make region, affinity scores using pseudo character-level GT bounding box
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """
        (image, word_level_char_bbox, words, confidence_mask, horizontal_text_bools) = self.load_data(index)
        img_h, img_w, _ = image.shape

        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
            all_affinity_bbox = []
        else:
            region_score = self.gaussian_builder.generate_region(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )
            affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )
























