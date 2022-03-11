
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import RandomResizedCrop, RandomCrop
from torchvision.transforms import InterpolationMode

def rescale_ic15(img, bboxes, target_size=2240):
    h, w = img.shape[0:2]
    scale = target_size / max(h,w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    bboxes = bboxes * scale
    return img, bboxes


def random_resize_crop_synth(augment_targets, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    short_side = min(image.size)
    i,j,h,w = RandomCrop.get_params(image, output_size=(short_side,short_side))

    image = resized_crop(image, i, j, h, w, size=(size, size))
    region_score = resized_crop(region_score, i, j, h, w, (size, size))
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size))
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets

def random_resize_crop(augment_targets, scale, ratio, size, threshold):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    if random.random() < threshold:
        i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    else:
        i, j, h, w = RandomResizedCrop.get_params(image, scale=(1.0,1.0), ratio=(1.0,1.0))

    image = resized_crop(image, i, j, h, w, size=(size, size))
    region_score = resized_crop(region_score, i, j, h, w, (size, size))
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size))
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets


def random_crop_with_bbox(augment_targets, word_level_char_bbox, output_size):
    h, w = augment_targets[0].shape[0:2]
    th, tw = output_size, output_size
    crop_h, crop_w = output_size, output_size
    if w == tw and h == th:
        return augment_targets

    word_bboxes = []
    if len(word_level_char_bbox) > 0:
        for bboxes in word_level_char_bbox:
             word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]

        left = max(sample_bboxes[1, 0] - output_size, 0)
        top = max(sample_bboxes[1, 1] - output_size,0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        # i, j = 0, 0
        # crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(augment_targets)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(augment_targets[idx].shape) == 3:
            augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            augment_targets[idx] = padding_image(augment_targets[idx], tw)


    return augment_targets

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_scale(images, word_level_char_bbox, scale_range):
    scale = random.sample(scale_range, 1)[0]

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], dsize=None, fx=scale, fy=scale)

    for i in range(len(word_level_char_bbox)):
        word_level_char_bbox[i] *= scale

    return images

def random_rotate(images, max_angle):
    angle = random.random() * 2 * max_angle - max_angle

    for i in range(len(images)):
        img = images[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        images[i] = img_rotation
    return images


def padding_image(image,imgsize):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img


def random_crop_v0(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
             word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]

        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0],0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        # i, j = 0, 0
        # crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs

def random_crop_v2(imgs, img_size):

    h, w = imgs[0].shape[0:2]
    th, tw = img_size

    i = random.randint(0, h-th)
    j = random.randint(0, w-tw)
    crop_h, crop_w = img_size

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs



# random crop cite from PaddleOCR
def is_poly_in_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
        return False
    if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
        return False
    return True


def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax

# cite from PaddleOCR
def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    h, w = im.shape[:2]
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in text_polys:
        for point in points:
            point = np.round(point, decimals=0).astype(np.int32)
            minx = np.min(point[:, 0])
            maxx = np.max(point[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(point[:, 1])
            maxy = np.max(point[:, 1])
            h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            ymin, ymax = random_select(h_axis, h)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        num_poly_in_rect = 0
        for polys in text_polys:
            for poly in polys:
                if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                            ymax - ymin):
                    num_poly_in_rect += 1
                    break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h

# cite from PaddleOCR
class EastRandomCropData(object):
    def __init__(self,
                 size=(640, 640),
                 max_tries=10,
                 min_crop_side_ratio=0.1,
                 keep_ratio=True,
                 **kwargs):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, region_scores, affinity_scores, text_polys):
        region_scores = region_scores
        affinity_scores = affinity_scores
        text_polys = text_polys

        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = crop_area(
            region_scores, text_polys, self.min_crop_side_ratio, self.max_tries)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg_region = np.zeros((self.size[1], self.size[0]),
                                 region_scores.dtype)
            padimg_region[:h, :w] = cv2.resize(
                region_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            region_scores = padimg_region

            padimg_affinity = np.zeros((self.size[1], self.size[0]),
                              affinity_scores.dtype)
            padimg_affinity[:h, :w] = cv2.resize(
                affinity_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            affinity_scores = padimg_affinity

        else:
            region_scores = cv2.resize(
                region_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
            affinity_scores = cv2.resize(
                affinity_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        for poly in zip(text_polys):
            poly = np.array([poly[i]-(crop_x,crop_y) for i in range(len(poly))])[0]
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
        return region_scores, affinity_scores, text_polys_crop