import os
import re
import itertools
import copy

import cv2
import h5py
import numpy as np
from PIL import Image
import scipy.io as scio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from config.load_config import load_yaml, DotDict
from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import (
    random_crop_with_bbox_adapt_to_output_size,
    random_horizontal_flip,
    random_rotate,
    random_scale,
    random_resize_crop,
)
from data.pseudo_label.make_charbox import make_pseudo_char_box
from utils.util import saveInput, saveImage


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
                saved_gt_dir=config.data_dir.synthtext_gt,
                gauss_init_size=config.train.data.gauss_init_size,
                gauss_sigma=config.train.data.gauss_sigma,
                enlarge_size=config.train.data.enlarge_size,
                aug=config.train.data.aug,
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
        saved_gt_dir,
        gauss_init_size,
        gauss_sigma,
        enlarge_size,
        aug,
        vis_opt,
    ):

        self.output_size = output_size
        self.data_dir = data_dir
        #self.gt = self.load_data(data_dir)


        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_size
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
        charBB = gt.attrs['charBB']
        txt = gt.attrs['txt']
        imgpath = gt.attrs['imgpath']

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
            img_h, img_w, word_level_char_bbox
        )
        affinity_score, _ = self.gaussian_builder.generate_affinity(
            img_h, img_w, word_level_char_bbox
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

        saveInput(
            self.img_names[index],
            image,
            region_score,
            affinity_score,
            confidence_mask,
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



class ICDAR2015(Dataset):
    def __init__(self, net, output_size, data_dir, saved_gt_dir, gauss_init_size, gauss_sigma, enlarge_size,
                 watershed_ver, aug, vis_opt, pseudo_vis_opt):

        self.net = net
        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.gaussian_builder = GaussianBuilder(gauss_init_size, gauss_sigma, enlarge_size)
        self.watershed_ver = watershed_ver
        self.aug = aug
        self.vis_opt = vis_opt
        self.pseudo_vis_opt = pseudo_vis_opt
        # self.vis_index = [189, 41, 723, 251, 232, 115, 634, 951, 247, 25, 400, 704, 619, 305, 423, 20, 31]
        self.vis_index = list(range(1000))
        self.img_dir = os.path.join(data_dir, 'ch4_training_images')
        self.img_gt_box_dir = os.path.join(data_dir, 'ch4_training_localization_transcription_gt')
        self.img_names = os.listdir(self.img_dir)

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
            word_bboxes.append(np.array(box_points).astype(np.float64))
            words.append(word)
        return word_bboxes, words

    def get_confidence_by_contour(self, image, region_score, word_bbox, word, new_imagename, vis=False):

        word_image, _ = self.crop_image_by_bbox(image, word_bbox)
        word_region_score, MM = self.crop_image_by_bbox(region_score, word_bbox)

        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_region_score.copy()
        # 왜 64로 scale 조절을 하는 걸까?? --> https://github.com/clovaai/CRAFT-pytorch/issues/18
        scale = 64.0 / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)

        ret, binary = cv2.threshold(input, 0.6 * 255, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        confidence = self.get_confidence(real_char_nums, len(contours))

        if confidence <= 0.5:  # confidence 값들이 낮은 경우, confidence 0.5
            confidence = 0.5

        if vis:
            word_image = cv2.resize(word_image, None, fx=scale, fy=scale)
            word_region_score = cv2.resize(word_region_score, None, fx=scale, fy=scale)
            word_region_score = cv2.applyColorMap(np.uint8(word_region_score), cv2.COLORMAP_JET)

            word_region_score = word_region_score.copy()
            binary = binary.copy()
            word_region_score = cv2.cvtColor(word_region_score, cv2.COLOR_BGR2RGB)
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # import ipdb; ipdb.set_trace()
            vis_result = np.hstack([word_image, word_region_score, binary])
            cv2.imwrite(f'/nas/home/gmuffiness/workspace/ocr_related/daintlab-CRAFT-Reimplementation/craft_jm/results_dir/exp_official_craft_supervision_v1.2/contour_sample/{new_imagename}_{confidence}.jpg', vis_result)
        return confidence


    def load_image_gt_and_confidence_mask(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0])
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)
        word_bboxes = np.float32(word_bboxes)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox = []
        new_words = []
        new_imagename = ''

        if len(word_bboxes) == 0:
            return image, word_level_char_bbox, new_words, confidence_mask

        for i in range(len(word_bboxes)):

            if self.pseudo_vis_opt and int(img_name.split('.')[0].split('_')[1]) in self.vis_index:
                new_imagename = img_name.split('.')[0] + '_' + str(i)

            pseudo_char_bbox, confidence = make_pseudo_char_box(self.net,
                                                                image,
                                                                word_bboxes[i],
                                                                words[i],
                                                                self.watershed_ver,
                                                                pseudo_vis_opt=self.pseudo_vis_opt,
                                                                img_name=new_imagename)

            # TODO: fill confidence mask 할 때, 더 낮은 값이 들어가도록 수정?
            if words[i] == '###' or len(words[i].strip()) == 0:
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                continue
            cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], confidence)
            new_words.append(words[i])
            word_level_char_bbox.append(pseudo_char_bbox)

        # TODO: new_words랑 words랑 다른지 확인

        return image, word_level_char_bbox, new_words, confidence_mask


    def load_image_gt_and_saved_confidence_mask(self, index):
        pass
        #-------------------------------------------------------------------------------------#

        # To make confidence_mask : 처음에만 실행될, save 할 confidence mask를 만드는 과정

        # confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        #
        # confidences = []
        # new_imagename = ''
        #
        # if len(word_bboxes) > 0:
        #     for i in range(len(word_bboxes)):
        #
        #         if words[i] == '###' or len(words[i].strip()) == 0:
        #             cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
        #             continue
        #         assert words[i] != '###' and len(words[i].strip()) != 0
        #
        #
        #         self.pseudo_vis_opt = False
        #         if int(img_name.split('.')[0].split('_')[1]) in self.vis_index :
        #             self.pseudo_vis_opt = True
        #             new_imagename = img_name.split('.')[0] +'_'+str(i)
        #
        #         query_idx = int(self.img_names[index].split('.')[0].split('_')[1])
        #         saved_region_scores_path = os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_region.jpg')
        #         # import ipdb; ipdb.set_trace()
        #         region_score = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        #         region_score = cv2.resize(region_score, (image.shape[1], image.shape[0])).astype(np.float32)
        #
        #         confidence = self.get_confidence_by_contour(image, region_score, word_bboxes[i], words[i], new_imagename, self.pseudo_vis_opt)
        #         confidences.append(confidence)
        #         cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
        #
        # return image, word_bboxes, confidence_mask, confidences

    # Save confidence_mask
    def save_confidence_mask(self, query_idx, confidence_mask):
        confidence_mask_copy = (confidence_mask * 255).astype(np.uint8)
        confidence_mask_copy = cv2.applyColorMap(confidence_mask_copy, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_cf_mask_jet_thresh_0.6.jpg'), confidence_mask_copy)
        cv2.imwrite(os.path.join(self.saved_gt_dir, f'res_img_{query_idx}_cf_mask_thresh_0.6.jpg'), confidence_mask)

    def make_pseudo_gt(self, index):
        image, word_level_char_bbox, words, confidence_mask = self.load_image_gt_and_confidence_mask(index)
        img_h, img_w, _ = image.shape

        if len(word_level_char_bbox) > 0:
            region_score = self.gaussian_builder.generate_region(img_h, img_w, word_level_char_bbox)
            affinity_score, _ = self.gaussian_builder.generate_affinity(img_h, img_w, word_level_char_bbox)
        else:
            region_score = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            affinity_score = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if int(self.img_names[index].split('.')[0].split('_')[1]) in self.vis_index:
            self.vis_opt = True

        return image, region_score, affinity_score, confidence_mask, word_level_char_bbox, words


    def load_saved_gt(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0])
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)
        word_bboxes = np.float32(word_bboxes)

        query_idx = int(self.img_names[index].split('.')[0].split('_')[1])

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

        # 기존 code 중 아래 random_crop에서 쓰이게 될 character bboxes 형식을 맞춰주기 위해, word bboxes를 1개의 character씩 담긴 bboxes로 만들어 줌
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

        #check minus coordinate
        for cb in word_level_char_bbox :
            if (cb < 0).astype('float32').sum() > 0 :
                import ipdb;ipdb.set_trace()
                print(query_idx)

        if int(self.img_names[index].split('.')[0].split('_')[1]) in self.vis_index and \
                self.img_names[index].split('_')[0] == 'img':
            self.vis_opt = True

        self.vis_opt = False

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
            augment_targets, word_level_char_bbox = random_scale(augment_targets, word_level_char_bbox, self.aug.random_scale.range)

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

        # query_idx = int(self.img_names[index].split('.')[0].split('_')[1])
        # print(self.vis_opt, query_idx)
        # # NOTE : 임시 test용으로만 사용할 코드라, 이미지 저장할 폴더 경로 hard-coding 되어 있음.
        # if self.vis_opt and query_idx in self.vis_index:
        #     saveImage(self.img_names[index], '/nas/home/gmuffiness/result/debug', image.copy(), word_level_char_bbox.copy(),
        #               region_score.copy(), affinity_score.copy(), confidence_mask.copy())
        #
        # image, region_score, affinity_score, confidence_mask = self.augment_image(
        #     image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        # )
        #
        # if self.vis_opt and query_idx in self.vis_index:
        #     saveInput(
        #         self.img_names[index],
        #         '/nas/home/gmuffiness/result/debug',
        #         image,
        #         region_score,
        #         affinity_score,
        #         confidence_mask,
        #     )

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




def test():
    import torch
    yaml_path = 'syn_test6_26-1'
    data_path =  "/nas/datahub/SynthText-KR"

    config = load_yaml(yaml_path)
    config = DotDict(config)
    dataloader = hierarchical_dataset(root=data_path, config=config)
    dataloader = ConcatDataset(dataloader)

    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True)


    total = 0
    for index, (image, region_scores, affinity_scores, confidence_mask) in enumerate(train_loader):
        total += 1
        print('$'*50)
        if total == 50:
            import ipdb;
            ipdb.set_trace()




