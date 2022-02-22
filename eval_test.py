# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import wandb
import yaml

from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.inference_boxes import test_net, load_icdar2015_gt, load_icdar2013_gt, load_synthtext_gt



def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def saveResult_2015(img_file, img, boxes, dirname='./result/', gt_file=None ):

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

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))


    for i, box in enumerate(boxes):

        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        except:
            pass

    if gt_file is not None:

        gt_name = "gt_" + filename + '.txt'

        with open(os.path.join(gt_file, gt_name), 'r', encoding="utf8", errors='ignore') as d:
            for l in d.read().splitlines():
                box = l.split(',')
                box_gt = np.array(list(map(int, box[:8])))
                gt_poly = box_gt.reshape(-1, 2)
                gt_poly = np.array(gt_poly).astype(np.int32)

                if box[-1] == '###':
                    cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(128, 128, 128), thickness=2)
                else:
                    cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

    # Save result image
    res_img_path = dirname + "/res_" + filename + '.jpg'
    cv2.imwrite(res_img_path, img)



def saveResult_2013(img_file, img, boxes, dirname='./result/', gt_file=None):
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

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(boxes):

        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        except:
            pass

    if gt_file is not None:

        gt_name = "gt_" + filename + '.txt'

        with open(os.path.join(gt_file, gt_name), 'r', encoding="utf8", errors='ignore') as d:
            for l in d.read().splitlines():
                box = l.split(',')
                box = [int(box[j]) for j in range(4)]
                box_gt = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])

                gt_poly = box_gt.reshape(-1, 2)
                gt_poly = np.array(gt_poly).astype(np.int32)

                cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


    # Save result image
    res_img_path = dirname + "/res_" + filename + '.jpg'
    cv2.imwrite(res_img_path, img)


def main(model_path, args, evaluator, data_li=''):


    # load net
    #------------------------------------------------------------------------------------------------------------------#
    model = CRAFT()  # initialize
    wandb.watch(model)
    # net = UNetWithResnet50Encoder()
    print('Loading weights from checkpoint (' + model_path + ')')
    net_param = torch.load(model_path)

    try:
        model.load_state_dict(copyStateDict(net_param['craft']))
    except:
        model.load_state_dict(copyStateDict(net_param))

    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = False
    model.eval()
    # print('Model setting completed.')
    # ------------------------------------------------------------------------------------------------------------------#


    if data_li != '':
        total_imgs_bboxes_gt, total_img_path = load_synthtext_gt(args.synthData_dir, data_li=data_li)

    else:
        test_folder = args.test_folder

        if test_folder.split('/')[-1].lower() == 'icdar2013':
            total_imgs_bboxes_gt, total_img_path, gt_folder_path = load_icdar2013_gt(dataFolder=test_folder,
                                                                     isTraing=args.isTraingDataset)
        else:
            total_imgs_bboxes_gt, total_img_path, gt_folder_path = load_icdar2015_gt(dataFolder=test_folder,
                                                                     isTraing=args.isTraingDataset)

    # print('icdar2015 data setting completed.')
    # ------------------------------------------------------------------------------------------------------------------#


    total_img_bboxes_pre = []
    for k, img_path in enumerate(tqdm(total_img_path)):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = imgproc.loadImage(img_path)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(model,
                                             image,
                                             args.text_threshold,
                                             args.link_threshold,
                                             args.low_text,
                                             args.cuda,
                                             args.poly,
                                             args.canvas_size,
                                             args.mag_ratio)

    # ------------------------------------------------------------------------------------------------------------------#
        if test_folder.split('/')[-1].lower() == 'icdar2013':
            rnd_list = [136, 210,  64,  97, 209,  87,  91, 169, 173, 191,  89, 177,  62,
                        105, 124, 213,  207, 216, 217,  34, 187,  42, 102, 113, 111, 176, 182, 1, 5, 8 ]
        else:
            rnd_list = [1, 264, 135, 352, 481, 250, 355, 436, 45, 181, 98, 173, 267, 200, 79, 395,
                        399, 162, 184, 217, 327, 344, 11, 107, 299, 244, 271, 92, 149, 259]


        viz = True
        if k in rnd_list:
            viz = True

        for box in bboxes:
            box_info = {"points": None, "text": None, "ignore": None}
            box_info["points"] = box
            box_info["text"] = "###"
            box_info["ignore"] = False
            single_img_bbox.append(box_info)
        total_img_bboxes_pre.append(single_img_bbox)

    # ----------------------------------------------------------------------------------------------------------------- #

        # if viz == True:
        #
        #     result_folder_name = (args.trained_model).split('/')[-2] + '_test_output_aligned_official_hp_setting'
        #
        #     outpath = os.path.join(os.path.join(args.results_dir, result_folder_name), str(utils.config.ITER))
        #     if not os.path.exists(outpath):
        #         os.makedirs(outpath)
        #
        #     if test_folder.split('/')[-1].lower() == 'icdar2013':
        #         saveResult_2013(img_path, image[:, :, ::-1].copy(), polys, dirname=outpath, gt_file=gt_folder_path)
        #     else:
        #         saveResult_2015(img_path, image[:, :, ::-1].copy(), polys, dirname=outpath, gt_file=gt_folder_path)
        #
        #
        #
        #     height, width, channel = image.shape
        #
        #     # ========== To save training image's region score with Official pretrained model ==========================
        #
        #     # if score_text[0].shape != (544, 960):
        #     #     import ipdb; ipdb.set_trace()
        #     # score_text[0] = score_text[0][:-4,:]
        #     # score_text[1] = score_text[1][:-4,:]
        #
        #     # ==========================================================================================================
        #     overlay_region = cv2.resize(score_text[0], (width, height))
        #     overlay_aff = cv2.resize(score_text[1], (width, height))
        #
        #     overlay_region = cv2.addWeighted(image.copy(), 0.4, overlay_region, 0.6, 5)
        #     overlay_aff = cv2.addWeighted(image.copy(), 0.4, overlay_aff, 0.6, 5)
        #
        #     # save overlay
        #     filename, file_ext = os.path.splitext(os.path.basename(img_path))
        #     overlay_region_file = outpath + "/res_" + filename + '_region.jpg'
        #     # cv2.imwrite(overlay_region_file, overlay_region)
        #
        #     filename, file_ext = os.path.splitext(os.path.basename(img_path))
        #     overlay_aff_file = outpath + "/res_" + filename + '_affi.jpg'
        #     # cv2.imwrite(overlay_aff_file, overlay_aff)
        #
        #     ori_image_path = outpath + "/res_" + filename + '.jpg'
        #     # cv2.imwrite(ori_image_path,image)
        #
        #     boxed_img = image.copy()
        #     for word_box in single_img_bbox:
        #         # sp = np.clip(np.min(word_box['points'], axis=0), 0, max(height, width)).astype(np.uint32)
        #         # ep = np.max(word_box['points'], axis=0).astype(np.uint32)
        #         # cv2.rectangle(boxed_img, sp, ep, (0, 0, 255), 3)
        #         # import ipdb;ipdb.set_trace()
        #         cv2.polylines(boxed_img, [word_box['points'].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=3)
        #
        #     box_image_path = outpath + "/res_" + filename + '_box.jpg'
        #     cv2.imwrite(box_image_path, boxed_img)
        #
        #     temp1 = np.hstack([image, boxed_img])
        #     temp2 = np.hstack([overlay_region, overlay_aff])
        #     temp3 = np.vstack([temp1, temp2])
        #
        #     cv2.imwrite(box_image_path, temp3)
        # # # --------------------------------------------------------------------------------------------------------#
    # print('Predict bbox points completed.')

    results = []
    for gt, pred in zip(total_imgs_bboxes_gt, total_img_bboxes_pre):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)

    wandb.log({"precision": metrics['precision'], "recall": metrics['recall'], "hmean": metrics['hmean']})
    return metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection Eval')
    parser.add_argument('--yaml_path', default='./exp/synthtext/', type=str, help='Load configuration')
    args = parser.parse_args()

    # load configure
    config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)

    # wandb
    wandb.init(project="jm-test", entity="pingu", name=res_dir_name)
    wandb.run.name = args.trained_model.split('/')[-2][-4:] + '_eval'
    wandb.config.update(config)


    evaluator = DetectionIoUEvaluator()
    main(args.trained_model, args, evaluator)
