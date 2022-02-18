import os
import cv2
import numpy as np
import time

import torch
from torch.autograd import Variable

#from utils import craft_utils
from data import imgproc
from collections import Iterable
import config
import craft_utils




def saveInput(imagename, image, region_scores, affinity_scores, confidence_mask):
    image = np.uint8(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes, polys = craft_utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.85, 0.2, 0.5, False)
    boxes = np.array(boxes, np.int32) * 2
    if len(boxes) > 0:
        np.clip(boxes[:, :, 0], 0, image.shape[1])
        np.clip(boxes[:, :, 1], 0, image.shape[0])
        for box in boxes:
            cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask / 255)

    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))
    confidence_mask_gray = cv2.resize(confidence_mask_gray, (width, height))

    overlay_region = cv2.addWeighted(image, 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image, 0.4, overlay_aff, 0.7, 6)

    gt_scores = np.concatenate([overlay_region, overlay_aff], axis=1)
    confidence_mask_gray = np.concatenate([np.zeros_like(confidence_mask_gray), confidence_mask_gray], axis=1)

    output = np.concatenate([gt_scores, confidence_mask_gray], axis=1)

    output = np.hstack([image, output])

    outpath = os.path.join(os.path.join(config.RESULT_DIR, '{}/input'.format(str(config.ITER // 100))),
                           "%s_input.jpg" % imagename)
    #print(outpath)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    cv2.imwrite(outpath, output)


def saveImage(imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
    output_image = np.uint8(image.copy())
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    if len(bboxes) > 0:
        affinity_bboxes = np.int32(affinity_bboxes)
        for i in range(affinity_bboxes.shape[0]):
            cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
        for i in range(len(bboxes)):
            _bboxes = np.int32(bboxes[i])
            for j in range(_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, overlay_aff, 0.6, 5)

    heat_map = np.concatenate([overlay_region, overlay_aff], axis=1)
    output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)

    outpath = os.path.join(os.path.join(config.RESULT_DIR, '{}/input'.format(str(config.ITER // 100))), imagename)
    #print(outpath)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    cv2.imwrite(outpath, output)




def save_parser(args):

    """ final options """
    with open(f'{args.results_dir}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)


def make_logger(path=False):

    # mode = iter or epoch

    def logger_path(path):

        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')

        trn_logger_path = os.path.join(f'{path}', f'train.log')
        val_logger_path = os.path.join(f'{path}', f'validation.log')


        return trn_logger_path, val_logger_path


    trn_logger_path, val_logger_path = logger_path( path)

    trn_logger = Logger(trn_logger_path)
    val_logger = Logger(val_logger_path)

    return trn_logger, val_logger


def split_logger(lang_dict):
    eopch_li = []
    loss_li = []
    #acc_li = []
    #ned_li = []

    for i in lang_dict:
        new_dict = i.split()
        eopch_li.append(int(new_dict[0]))
        loss_li.append(float(new_dict[1]))
        #acc_li.append(float(new_dict[2]))
        #ned_li.append(float(new_dict[3]))

    return eopch_li, loss_li

def read_txt(path):
    with open(path, 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]

    eopch_li, loss_li = split_logger(lang_dict)

    return eopch_li, loss_li

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log





class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)

        else:
            pass