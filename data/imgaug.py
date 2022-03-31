
import random

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resized_crop, crop
from torchvision.transforms import RandomResizedCrop, RandomCrop
from torchvision.transforms import InterpolationMode

def rescale_ic15(img, bboxes, target_size=2240):
    h, w = img.shape[0:2]
    scale = target_size / max(h,w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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

    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets


def random_resize_crop_ai(augment_targets, scale, ratio, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)


    i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)

    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
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


    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)


    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets


def random_crop(augment_targets, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)


    i,j,h,w = RandomCrop.get_params(image, output_size=(size,size))

    image = crop(image, i, j, h, w)
    region_score = crop(region_score, i, j, h, w)
    affinity_score = crop(affinity_score, i, j, h, w)
    confidence_mask = crop(confidence_mask, i, j, h, w)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets





def random_crop_with_bbox(augment_targets, word_level_char_bbox, output_size):
    # h, w = augment_targets[0].shape[0:2]
    # th, tw = output_size, output_size
    # crop_h, crop_w = output_size, output_size
    # if w == tw and h == th:
    #     return augment_targets

    # word_bboxes = []
    # if len(word_level_char_bbox) > 0:
    #     for bboxes in word_level_char_bbox:
    #          word_bboxes.append(
    #             [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    # word_bboxes = np.array(word_bboxes, np.int32)

    # if random.random() > 0.6 and len(word_bboxes) > 0:
    #     sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]

    #     left = max(sample_bboxes[1, 0] - output_size, 0)
    #     top = max(sample_bboxes[1, 1] - output_size,0)

    #     if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
    #         i = random.randint(0, h - th)
    #         j = random.randint(0, w - tw)
    #     else:
    #         i = random.randint(top, min(sample_bboxes[0, 1], h - th))
    #         j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

    #     crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
    #     crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    # else:
    #     ### train for IC15 dataset####
    #     i = random.randint(0, h - th)
    #     j = random.randint(0, w - tw)

    #     #### train for MLT dataset ###
    #     # i, j = 0, 0
    #     # crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    # for idx in range(len(augment_targets)):
    #     # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
    #     # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

    #     if len(augment_targets[idx].shape) == 3:
    #         augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w, :]
    #     else:
    #         augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w]

    #     if crop_w > tw or crop_h > th:
    #         augment_targets[idx] = padding_image(augment_targets[idx], tw)


    # return augment_targets
    pass

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
        if i == len(images) - 1:
            img_rotation = cv2.warpAffine(img, M=rotation_matrix, dsize=(h, w), flags=cv2.INTER_NEAREST)
        else:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        images[i] = img_rotation
    return images

