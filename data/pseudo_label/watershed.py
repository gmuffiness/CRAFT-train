import time
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from data import imgproc
import Polygon as plg

from collections import OrderedDict
from data.boxEnlarge import enlargebox

def watershed_v2(region_score, input_img, pseudo_vis_opt):

    if region_score.max() < 255 * 0.05:
        return np.array([], dtype=np.uint8), np.zeros(region_score.shape, np.uint8)

    ori_input_img = input_img.copy()
    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_score

    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, init_markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    init_markers = init_markers + 1
    # Now, mark the region of unknown with zero
    init_markers[unknown == 255] = 0
    init_markers_copy = init_markers.copy()

    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX) * 255
    dist_transform = np.uint8(dist_transform)
    dist_transform = cv2.cvtColor(dist_transform, cv2.COLOR_GRAY2RGB)
    ret, dist_transform_binary = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    final_markers = cv2.watershed(dist_transform_binary, init_markers)
    region_score[final_markers == -1] = [255, 0, 0]

    color_markers = np.uint8(final_markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    # make boxes
    boxes = []
    for i in range(2, np.max(final_markers) + 1):

        # 변경 후 : make box without angle
        try:
            x_min, x_max = np.min(np.where(final_markers == i)[1]), np.max(np.where(final_markers == i)[1])
            y_min, y_max = np.min(np.where(final_markers == i)[0]), np.max(np.where(final_markers == i)[0])
            # print(x_min, x_max, y_min, y_max)
            box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            # cv2.polylines(input_img, [np.array(box, dtype=np.int)], True, (0, 0, 255), 5)

            # 변경 전 : make box with angle(minAreaRect)

            # np_contours = np.roll(np.array(np.where(final_markers == i)), 1, axis=0).transpose().reshape(-1, 2)
            # rectangle = cv2.minAreaRect(np_contours)
            # box = cv2.boxPoints(rectangle)

            # startidx = box.sum(axis=1).argmin()
            # box = np.roll(box, 4 - startidx, 0)
            # poly = plg.Polygon(box)
            # area = poly.area()
            # if area < 10:
            #     continue

            box = np.array(box)
            boxes.append(box)
        except:
            sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
            sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
            unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

            init_markers_copy = np.uint8(init_markers_copy + 1)
            init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
            init_markers_copy = np.uint8(init_markers_copy)
            init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)

            region_score = cv2.applyColorMap(region_score, cv2.COLORMAP_JET)

            vis_result = np.vstack(
                [ori_input_img, ori_region_score, sure_bg_copy, dist_transform, sure_fg_copy, unknown_copy,
                 init_markers_copy, dist_transform_binary,
                 color_markers, region_score, input_img])
            cv2.imwrite('./results_dir/exp_v2.2/watershed/{}'.format(f'watershed_result_{random.random()}.png'),
                        vis_result)

    #boxes = np.array(boxes) * 2
    #boxes = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))

    if pseudo_vis_opt:
        sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
        sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
        unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

        init_markers_copy = np.uint8(init_markers_copy + 1)
        init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
        init_markers_copy = np.uint8(init_markers_copy)
        init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)

        region_score =  cv2.applyColorMap(region_score, cv2.COLORMAP_JET)

        vis_result = np.vstack(
            [ori_input_img, ori_region_score, sure_bg_copy, dist_transform, sure_fg_copy, unknown_copy, init_markers_copy, dist_transform_binary,
             color_markers, region_score, input_img])
        cv2.imwrite('./results_dir/exp_v2.1/watershed/{}'.format(f'watershed_result_{random.random()}.png'), vis_result)

    # import ipdb; ipdb.set_trace()
    return np.array(boxes), color_markers

def watershed_v3(region_score, input_img, pseudo_vis_opt):

    if region_score.max() < 255 * 0.5:
        return np.array([], dtype=np.uint8), np.zeros(region_score.shape, np.uint8)

    ori_input_img = input_img.copy()
    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_score

    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, init_markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    init_markers = init_markers + 1
    # Now, mark the region of unknown with zero
    init_markers[unknown == 255] = 0
    init_markers_copy = init_markers.copy()

    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX) * 255
    dist_transform = np.uint8(dist_transform)
    dist_transform = cv2.cvtColor(dist_transform, cv2.COLOR_GRAY2RGB)
    ret, dist_transform_binary = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    final_markers = cv2.watershed(dist_transform_binary, init_markers)
    region_score[final_markers == -1] = [255, 0, 0]

    color_markers = np.uint8(final_markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    # make boxes
    boxes = []
    for i in range(2, np.max(final_markers) + 1):

        # 변경 후 : make box without angle
        try:
            x_min, x_max = np.min(np.where(final_markers == i)[1]), np.max(np.where(final_markers == i)[1])
            y_min, y_max = np.min(np.where(final_markers == i)[0]), np.max(np.where(final_markers == i)[0])
            # print(x_min, x_max, y_min, y_max)
            box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            # cv2.polylines(input_img, [np.array(box, dtype=np.int)], True, (0, 0, 255), 5)

            # 변경 전 : make box with angle(minAreaRect)

            # np_contours = np.roll(np.array(np.where(final_markers == i)), 1, axis=0).transpose().reshape(-1, 2)
            # rectangle = cv2.minAreaRect(np_contours)
            # box = cv2.boxPoints(rectangle)

            # startidx = box.sum(axis=1).argmin()
            # box = np.roll(box, 4 - startidx, 0)
            # poly = plg.Polygon(box)
            # area = poly.area()
            # if area < 10:
            #     continue

            box = np.array(box)
            boxes.append(box)
        except:
            sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
            sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
            unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

            init_markers_copy = np.uint8(init_markers_copy + 1)
            init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
            init_markers_copy = np.uint8(init_markers_copy)
            init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)

            region_score = cv2.applyColorMap(region_score, cv2.COLORMAP_JET)

            vis_result = np.vstack(
                [ori_input_img, ori_region_score, sure_bg_copy, dist_transform, sure_fg_copy, unknown_copy,
                 init_markers_copy, dist_transform_binary,
                 color_markers, region_score, input_img])
            cv2.imwrite('./results_dir/exp_v2.2/watershed/{}'.format(f'watershed_result_{random.random()}.png'),
                        vis_result)

    #boxes = np.array(boxes) * 2
    #boxes = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))

    if pseudo_vis_opt:
        sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
        sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
        unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

        init_markers_copy = np.uint8(init_markers_copy + 1)
        init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
        init_markers_copy = np.uint8(init_markers_copy)
        init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)

        region_score =  cv2.applyColorMap(region_score, cv2.COLORMAP_JET)

        vis_result = np.vstack(
            [ori_input_img, ori_region_score, sure_bg_copy, dist_transform, sure_fg_copy, unknown_copy, init_markers_copy, dist_transform_binary,
             color_markers, region_score, input_img])
        cv2.imwrite('./results_dir/exp_v2.1/watershed/{}'.format(f'watershed_result_{random.random()}.png'), vis_result)

    # import ipdb; ipdb.set_trace()
    return np.array(boxes), color_markers

def exec_watershed_by_version(watershed_ver, bgr_region_scores, input, pseudo_vis_opt):

    # 새로운 watershed version을 추가할 때마다, 아래 dict에 추가해줘야 함.
    # => 더 깔끔하게 할 수 없을까?
    func_name_map_dict = {
        2: watershed_v2,
        3: watershed_v3,
        # '4': watershed_v4,
    }

    try:
        return func_name_map_dict[watershed_ver](bgr_region_scores, input, pseudo_vis_opt)
    except:
        print(f'Watershed version {watershed_ver} does not exist in func_name_map_dict.')