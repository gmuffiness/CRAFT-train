import os
import re
import itertools

import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import random_scale, random_rotate, random_crop_with_bbox_adapt_to_output_size, random_resize_crop, random_horizontal_flip
from data.pointClockOrder import mep
from data.pseudo_label.watershed import exec_watershed_by_version

def crop_image_by_bbox(image, box):
    w = (int)(np.linalg.norm(box[0] - box[1]))
    h = (int)(np.linalg.norm(box[0] - box[3]))
    if h > w * 1.5:
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[w, 0], [w, h], [0, h], [0, 0]])))
    else:
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]])))

    warped = cv2.warpPerspective(image, M, (w, h))
    return warped, M


def get_confidence(real_len, pseudo_len):
    if pseudo_len == 0:
        return 0.
    return (real_len - min(real_len, abs(real_len - pseudo_len))) / real_len

def inference_word_box(net, word_image):
    # print(f'In GPU {torch.cuda.current_device()}')
    # print('Model last parameters : {}'.format(net.conv_cls[-1].weight.reshape(2, -1)))

    if net.training:
        net.eval()

    net = net.cuda()
    with torch.no_grad():
        word_img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(word_image, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        word_img_torch = word_img_torch.permute(2, 0, 1).unsqueeze(0)
        word_img_torch = word_img_torch.type(torch.FloatTensor).cuda()
        word_img_scores, _ = net(word_img_torch)

    net.train()
    return word_img_scores, net

def visualize_pseudo_label(word_image, region_score, pseudo_char_bbox, bboxes, color_markers, img_name):
    word_img_h, word_img_w, _ = word_image.shape

    input_copy1 = word_image.copy()
    _purs_bboxes = np.int32(pseudo_char_bbox.copy())
    if len(_purs_bboxes) > 0:
        _purs_bboxes[:, :, 0] = np.clip(_purs_bboxes[:, :, 0], 0, word_img_w)
        _purs_bboxes[:, :, 1] = np.clip(_purs_bboxes[:, :, 1], 0, word_img_h)
        for bbox_p in _purs_bboxes:
            cv2.polylines(np.uint8(input_copy1), [np.reshape(bbox_p, (-1, 1, 2))], True, (255, 0, 0))

    input_copy2 = word_image.copy()
    _tmp_bboxes = np.int32(bboxes.copy())
    _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, word_img_w)
    _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, word_img_h)
    for bbox in _tmp_bboxes:
        cv2.polylines(np.uint8(input_copy2), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))

    region_scores_color = cv2.applyColorMap(np.uint8(region_score), cv2.COLORMAP_JET)
    region_scores_color = cv2.resize(region_scores_color, (word_img_w, word_img_h))

    # viz_image2 = np.hstack([word_image[:, :, ::-1], region_scores_color, color_markers,
    #                        input_copy1[:, :, ::-1], input_copy2[:, :, ::-1]])
    # cv2.imwrite('/nas/home/gmuffiness/result/temp_hstack.jpg', viz_image2)

    # gaussian
    gaussian_builder = GaussianBuilder(200, 40, 0.5)
    target = gaussian_builder.generate_region(region_scores_color.shape[0], region_scores_color.shape[1], [_tmp_bboxes])
    target_color = cv2.applyColorMap(target.astype('uint8'), cv2.COLORMAP_JET)

    overlay_img = cv2.addWeighted(word_image[:, :, ::-1], 0.7, target_color, 0.3, 5)
    # ori img , region score, watershed, box img
    viz_image = np.hstack([word_image[:, :, ::-1], region_scores_color, color_markers,
                           input_copy1[:, :, ::-1], input_copy2[:, :, ::-1], target_color, overlay_img])

    save_path = os.path.join('/nas/home/gmuffiness/result/debug', str(0 // 100))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(os.path.join(save_path, '{}_{}'.format(img_name, 'hstack.jpg')), viz_image)
    # if config.ITER == 0:
    # cv2.imwrite(os.path.join(os.path.join(save_path, 'ori_img_v3'), '{}_{}'.format(img_name, 'img.jpg')), word_image[:, :, ::-1])
    # cv2.imwrite(os.path.join(os.path.join(save_path, 'region_score_v3'), '{}_{}'.format(img_name, 'region_score.jpg')), region_score_rgb)

    # import ipdb; ipdb.set_trace()

def exclude_small_char(pseudo_char_bbox):
    bbox = []
    # except for the small box
    for i in range(pseudo_char_bbox.shape[0]):
        if np.mean(pseudo_char_bbox[i].ravel()) > 2:  # ravel -> 1차원 변환
            bbox.append(pseudo_char_bbox[i])
        else:
            print("filter bboxes", pseudo_char_bbox[i])  # 작은 box들

        # check small box 2
        # import ipdb;ipdb.set_trace()
        #
        # poly = plg.Polygon(pseudo_char_bbox[i])
        # area = poly.area()
        # if area < 10:
        #     continue
        # bbox.append(pseudo_char_bbox[i])
        # pursedo_bboxes_ = pseudo_char_bbox[i].copy()
        # top_left = np.array([np.min(pseudo_char_bbox[i][:, 0]), np.min(pseudo_char_bbox[i][:, 1])]).astype(np.int32)
        # pursedo_bboxes_ -= top_left[None, :]
        #
        # width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
        #     pursedo_bboxes_[:, 1]).astype(np.int32)
        #
        # if width >0 or height >0:
        #     bbox.append(pseudo_char_bbox[i])
        # else:
        #     import ipdb;ipdb.set_trace()
        #     print("filter bboxes", pseudo_char_bbox[i])  # 작은 box들

        # for cb in bboxes:
        #     # if (cb < 0).astype('float32').sum() > 0:
        #     #     import ipdb;
        #
        #     #check 1
        #     poly = plg.Polygon(cb)
        #     area = poly.area()
        #     if area < 10:
        #         import ipdb;ipdb.set_trace()
        #
        #     # check 2
        #     pursedo_bboxes_ = cb.copy()
        #     top_left = np.array([np.min(pseudo_char_bbox[:, 0]), np.min(pseudo_char_bbox[:, 1])]).astype(np.int32)
        #     pursedo_bboxes_ -= top_left[None, :]
        #
        #     width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
        #         pursedo_bboxes_[:, 1]).astype(np.int32)
        #
        #     if width >0 or height >0:
        #         pass
        #     else:
        #         import ipdb;ipdb.set_trace()
        #         print("filter bboxes", pseudo_char_bbox[i])  # 작은 box들

    return bbox

def split_word_equal(word_img_w, word_img_h, word, bboxes):
    width = word_img_w
    height = word_img_h

    width_per_char = width / len(word)
    for j, char in enumerate(word):
        if char == ' ':
            continue
        left = j * width_per_char
        right = (j + 1) * width_per_char
        bbox = np.array([[left, 0], [right, 0], [right, height],
                         [left, height]])
        bboxes.append(bbox)

    bboxes = np.array(bboxes, np.float32)
    return bboxes

def make_pseudo_char_box(net, image, word_bbox, word, watershed_ver, pseudo_vis_opt=False, img_name=''):

    word_image, MM = crop_image_by_bbox(image, word_bbox)
    real_word_without_space = word.replace('\s', '')
    real_char_len = len(real_word_without_space)
    # Fix height to 64 --> https://github.com/clovaai/CRAFT-pytorch/issues/18
    scale = 64.0 / word_image.shape[0]
    word_image = cv2.resize(word_image, None, fx=scale, fy=scale)
    word_img_h, word_img_w, _ = word_image.shape

    scores, net = inference_word_box(net, word_image)
    region_score = scores[0, :, :, 0].cpu().data.numpy()
    region_score = np.uint8(np.clip(region_score, 0, 1) * 255)

    # TODO: resize를 안하고 watershed 하는 것과 뭐가 더 나을지?
    region_score_rgb = cv2.resize(region_score, (word_img_w, word_img_h))
    region_score_rgb = cv2.cvtColor(region_score_rgb, cv2.COLOR_GRAY2RGB)

    pseudo_char_bbox, color_markers = exec_watershed_by_version(watershed_ver, region_score_rgb, word_image, pseudo_vis_opt)

    if len(pseudo_char_bbox) > 0:
        pseudo_char_bbox[:, :, 0] = np.clip(pseudo_char_bbox[:, :, 0], 0, region_score_rgb.shape[1])
        pseudo_char_bbox[:, :, 1] = np.clip(pseudo_char_bbox[:, :, 1], 0, region_score_rgb.shape[0])

    pseudo_char_bbox = exclude_small_char(pseudo_char_bbox)
    pseudo_char_bbox = np.array(pseudo_char_bbox, np.float32)

    if pseudo_char_bbox.shape[0] > 1:
        import ipdb; ipdb.set_trace()
        index = np.argsort(pseudo_char_bbox[:, 0, 0])
        pseudo_char_bbox = pseudo_char_bbox[index]

    confidence = get_confidence(real_char_len, len(pseudo_char_bbox))

    bboxes = []
    if confidence <= 0.5:  # confidence 값들이 낮은 경우 등분하고, 이떄 confidence 0.5
        bboxes = split_word_equal(word_img_w, word_img_h, word, bboxes)
        confidence = 0.5
    else:
        bboxes = pseudo_char_bbox

    if pseudo_vis_opt:
        visualize_pseudo_label(word_image, region_score, pseudo_char_bbox, bboxes, color_markers, img_name)

    bboxes /= scale

    # NOTE: 이전의 box 좌표 inverse warping code => 만약 평행사변형이 아닐 경우, 이상한 곳으로 warping 되는 이슈
    # try:  # problem
    #     for k in range(len(bboxes)):
    #         ones = np.ones((4, 1))
    #         tmp = np.concatenate([bboxes[k], ones], axis=-1)
    #         I = np.matrix(MM).I
    #         ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
    #         bboxes[k] = ori[:, :2]
    # except Exception as e:
    #     print(e)

    # NOTE : new method of warping box points inversely
    inv_trans = np.linalg.pinv(MM)
    for i in range(len(bboxes)):
        bboxes[i] = cv2.perspectiveTransform(bboxes[i][None, :, :], inv_trans)

    # TODO: 함수로 만들기 => 빠져나간 좌표들 최소, 최대 사이즈로 복원
    if len(bboxes) > 0:
        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0, image.shape[0])
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0, image.shape[1])

    return bboxes, confidence