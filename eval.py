# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import wandb
import yaml

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.inference_boxes import test_net, load_icdar2015_gt, load_icdar2013_gt, load_synthtext_gt
from utils.util import copyStateDict



def save_result_synth(img_file, img, pre_output, pre_box, gt_box=None, result_dir=''):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))


    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        except:
            pass


    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(img, [np.array(gt_box[j]['points']).astype(np.int32).reshape((-1, 1, 2))],
                          True, color=(0, 0, 255), thickness=2)

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + '.jpg'
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + '_box.jpg'
    cv2.imwrite(overlay_image_path, overlay_img)




def save_result_2015(img_file, img, pre_output, pre_box, gt_box, result_dir):

    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        except:
            pass

    if gt_box is not None:
        for j in range(len(gt_box)):
            _gt_box = np.array(gt_box[j]["points"]).reshape(-1, 2).astype(np.int32)
            if gt_box[j]["text"] == "###":
                cv2.polylines(img, [_gt_box], True, color=(128, 128, 128), thickness=2)
            else:
                cv2.polylines(img, [_gt_box], True, color=(0, 0, 255), thickness=2)


    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + '.jpg'
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + '_box.jpg'
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2013(img_file, img, pre_output, pre_box, gt_box=None, result_dir=''):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))


    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        except:
            pass


    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(img, [np.array(gt_box[j]['points']).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + '.jpg'
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + '_box.jpg'
    cv2.imwrite(overlay_image_path, overlay_img)



def overlay(image, region, affinity, single_img_bbox):

    height, width, channel = image.shape

    region_score = cv2.resize(region, (width, height))
    affinity_score = cv2.resize(affinity, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, region_score, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, affinity_score, 0.6, 5)

    # draw
    boxed_img = image.copy()
    for word_box in single_img_bbox:
        cv2.polylines(boxed_img, [word_box.astype(np.int32).reshape((-1, 1, 2))], True,
                      color=(0, 255, 0),thickness=3)

    temp1 = np.hstack([image, boxed_img])
    temp2 = np.hstack([overlay_region, overlay_aff])
    temp3 = np.vstack([temp1, temp2])

    return temp3


def load_test_dataset(test_folder_name, config):
    #TODO if문을 삭제할 수 있지 않을까??


    if test_folder_name == 'synthtext':
        total_bboxes_gt, total_img_path = load_synthtext_gt(config.test.test_folder)

    elif test_folder_name == 'icdar2013':
            total_bboxes_gt, total_img_path = load_icdar2013_gt(dataFolder=config.test.test_folder,
                                                                isTraing=config.test.isTraingDataset)

    elif test_folder_name == 'icdar2015':
        total_bboxes_gt, total_img_path = load_icdar2015_gt(dataFolder=config.test.test_folder,
                                                            isTraing=config.test.isTraingDataset)

    else: print('not found test dataset')

    return total_bboxes_gt, total_img_path


def viz_test(img, pre_output, pre_box, gt_box, img_name, result_dir, test_folder_name):

    if test_folder_name == 'synthtext':
        save_result_synth(img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir)
    elif test_folder_name == 'icdar2013':
        save_result_2013(img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir)
    elif test_folder_name == 'icdar2015':
        save_result_2015(img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir)
    else:
        print('not found test dataset')




def main(model_path, config, evaluator, result_dir, viz=False):

    # test 폴더에 대한 학습된 모델의 f1-score를 계산
    # test 폴더에 대한 model의 output 시각화
    # TODO loss 까지 구할 수 있도록?

    # model_path : 학습된 모델의 저장 경로
    # config : test에 필요한 configuration, dict type
    # evaluator : test function

    test_folder_name = config.test.test_folder.split('/')[-2].lower()


    # load model
    model = CRAFT()  # initialize
    print('Loading weights from checkpoint (' + model_path + ')')
    net_param = torch.load(model_path)
    model.load_state_dict(copyStateDict(net_param['craft']))

    if config.test.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False

    model.eval()
    # ------------------------------------------------------------------------------------------------------------------#

    total_imgs_bboxes_gt, total_imgs_path  = load_test_dataset(test_folder_name, config)

    # ------------------------------------------------------------------------------------------------------------------#


    total_img_bboxes_pre = []
    for k, img_path in enumerate(tqdm(total_imgs_path)):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(model,
                                             image,
                                             config.test.text_threshold,
                                             config.test.link_threshold,
                                             config.test.low_text,
                                             config.test.cuda,
                                             config.test.poly,
                                             config.test.canvas_size,
                                             config.test.mag_ratio)

    # ------------------------------------------------------------------------------------------------------------------#

        for box in bboxes:
            box_info = {"points": None, "text": None, "ignore": None}
            box_info["points"] = box
            box_info["text"] = "###"
            box_info["ignore"] = False
            single_img_bbox.append(box_info)
        total_img_bboxes_pre.append(single_img_bbox)

    # ------------------------------------------------------------------------------------------------------------------#

        viz_test(image, score_text, pre_box=polys, gt_box=total_imgs_bboxes_gt[k],
                  img_name=img_path, result_dir=result_dir, test_folder_name=test_folder_name)

    # ------------------------------------------------------------------------------------------------------------------#

    # print('Predict bbox points completed.')
    results = []
    for gt, pred in zip(total_imgs_bboxes_gt, total_img_bboxes_pre):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)

    #wandb.log({"precision": metrics['precision'], "recall": metrics['recall'], "hmean": metrics['hmean']})
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection Eval')
    parser.add_argument('--yaml','--yaml_file_name', default='./exp/synthtext/', type=str, help='Load configuration')
    args = parser.parse_args()

    # load configure
    config = load_yaml(args.yaml)
    config = DotDict(config)

    # Make result_dir
    res_dir = os.path.join(os.path.join('exp', args.yaml),'result')
    config.results_dir = res_dir
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    # wandb
    #wandb.init(project="jm-test", entity="pingu", name=args.yaml)
    #wandb.config.update(config)

    evaluator = DetectionIoUEvaluator()
    main(config.test.trained_model, config, evaluator, res_dir)
