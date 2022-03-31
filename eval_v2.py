# -*- coding: utf-8 -*-

import argparse
import os

import json
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import wandb
import yaml

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from model.craft_resnet import UNetWithResnet50Encoder
from metrics.eval_det_iou import DetectionIoUEvaluator
from metrics.clEval import script as clEval
from utils.inference_boxes import (
    test_net,
    load_icdar2015_gt,
    load_icdar2013_gt,
    load_synthtext_gt,
    load_prescription_cleval_gt,
    load_prescription_gt
)
from utils.util import copyStateDict
from data import imgproc


# NOTE
def result_to_clEval(bounds):
    # To use PopEval metric, the output of EasyOCR should be reorganized into txt form.
    #
    # bounds 리스트를 받아서, 아래 예시와 같은 str 형식으로 바꿔주는 함수
    # ==============================================================
    # Example)
    # input :
    # [([[273, 33], [519, 33], [519, 73], [273, 73]],
    # '진료비 세부내역서',
    # 0.7085034251213074)]
    #
    # output :
    # 273,33,519,33,519,73,273,73,"진료비 세부내역서"
    # ===============================================================

    result = ''
    for i, bound_sample in enumerate(bounds):
        point_list = bound_sample
        point_list_flatten = [str(int(coordinate)) for point in point_list for
                              coordinate in point]

        point_str = ', '.join(point_list_flatten)
        output = point_str
        result = result + output + '\n'

    return result

# NOTE
def make_txt(result_str, save_folder, filename, dtype):
    # str문자열 받아서, 문자열을 txt형식 파일으로 만들어서 PATH에 저장하는 함수.

    if dtype == 'label':
        filename = filename + '_label'
    elif dtype == 'pred':
        filename = filename + '_pred'
    else:
        filename = filename + '_{}'.format(str(dtype))

    result_txt = os.path.join(
        save_folder, '{}.txt'.format(filename))

    text_file = open(result_txt, "w", encoding='utf8')
    text_file.write(result_str)
    text_file.close()

    #return print('save_successful')


def save_result_synth(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

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
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).astype(np.int32).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2015(img_file, img, pre_output, pre_box, gt_box, result_dir):

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
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
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
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)

    # rg_score_image = np.uint8(pre_output[0] * 255)
    # affi_score_image = np.uint8(pre_output[1] * 255)
    # score_image = np.hstack([rg_score_image, affi_score_image])
    # score_img_path = result_dir + "/res_" + filename + "_score_grayscale.jpg"
    # cv2.imwrite(score_img_path, score_image)


def save_result_2013(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

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
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def overlay(image, region, affinity, single_img_bbox):

    height, width, channel = image.shape

    region_score = cv2.resize(region, (width, height))
    affinity_score = cv2.resize(affinity, (width, height))

    #region_score_color = imgproc.cvt2HeatmapImg(region_score)
    #affinity_score_color = imgproc.cvt2HeatmapImg(affinity_score)

    overlay_region = cv2.addWeighted(image.copy(), 0.4, region_score, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, affinity_score, 0.6, 5)

    # draw
    boxed_img = image.copy()
    for word_box in single_img_bbox:
        cv2.polylines(
            boxed_img,
            [word_box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=3,
        )

    temp1 = np.hstack([image, boxed_img])
    temp2 = np.hstack([overlay_region, overlay_aff])
    temp3 = np.vstack([temp1, temp2])

    return temp3


def load_test_dataset_iou(test_folder_name, config):
    # TODO if문을 삭제할 수 있지 않을까??

    if test_folder_name == "synthtext":
        total_bboxes_gt, total_img_path = load_synthtext_gt(config.test_data_dir)

    elif test_folder_name == "icdar2013":
        total_bboxes_gt, total_img_path = load_icdar2013_gt(
            dataFolder=config.test_data_dir)

    elif test_folder_name == "icdar2015":
        total_bboxes_gt, total_img_path = load_icdar2015_gt(
            dataFolder=config.test_data_dir)
    # NOTE
    elif test_folder_name == "prescription":
        total_bboxes_gt, total_img_path = load_prescription_gt(
            dataFolder=config.test_data_dir)



    else:
        print("not found test dataset")

    return total_bboxes_gt, total_img_path


def load_test_dataset_cl(test_folder_name, config):
    # TODO if문을 삭제할 수 있지 않을까??

    if test_folder_name == "synthtext":
        total_bboxes_gt, total_img_path = load_synthtext_gt(config.test_data_dir)

    elif test_folder_name == "icdar2013":
        total_bboxes_gt, total_img_path = load_icdar2013_gt(
            dataFolder=config.test_data_dir)

    elif test_folder_name == "icdar2015":
        total_bboxes_gt, total_img_path = load_icdar2015_gt(
            dataFolder=config.test_data_dir)

    elif test_folder_name == "prescription":
        total_bboxes_gt, total_img_path = load_prescription_cleval_gt(
            dataFolder=config.test_data_dir)

    else:
        print("not found test dataset")

    return total_bboxes_gt, total_img_path




def viz_test(img, pre_output, pre_box, gt_box, img_name, result_dir, test_folder_name):

    if test_folder_name == "synthtext":
        save_result_synth(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    elif test_folder_name == "icdar2013":
        save_result_2013(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    elif test_folder_name == "icdar2015" or test_folder_name == "prescription" :
        save_result_2015(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    else:
        print("not found test dataset")


def load_gt_cl_dir(config, data):
    if data == "icdar2013":
        gt_cl_dir = os.path.join(config.test_data_dir, "Challenge2_Test_Task1_GT_cl")
    elif data == "icdar2015":
        gt_cl_dir = os.path.join(config.test_data_dir, "ch4_test_localization_transcription_gt_cl")
    elif data == "prescription":
        gt_cl_dir = config.test_data_dir
    else:
        gt_cl_dir = None
        print("no dataset")
    return gt_cl_dir


def main_eval(model_path, backbone, config, evaluator, result_dir, buffer, model):

    # test 폴더에 대한 학습된 모델의 f1-score를 계산
    # test 폴더에 대한 model의 output 시각화
    # TODO loss 까지 구할 수 있도록?

    # model_path : 학습된 모델의 저장 경로
    # config : test에 필요한 configuration, dict type
    # evaluator : test function

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # check buffer for distributed evaluation
    assert all(v is None for v in buffer), 'Buffer already filled with another value'
    # print(len(total_imgs_bboxes_gt)) # 500
    # print('Current cuda device:', torch.cuda.current_device())
    # print('Total gpu :', torch.cuda.device_count())
    gpu_idx = torch.cuda.current_device()
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)

    test_set = config.test_data_dir.split("/")[-2].lower()

    # load model
    # if backbone == "vgg":
    #     model = CRAFT()  # initialize
    # elif backbone == "resnet":
    #     model = UNetWithResnet50Encoder()
    # else:
    #     raise Exception('Undefined architecture')
    #
    # print("Loading weights from checkpoint (" + model_path + ")")
    # net_param = torch.load(model_path, map_location=f'cuda:{gpu_idx}')
    # model.load_state_dict(copyStateDict(net_param["craft"]))
    #
    # if config.cuda:
    #     model = model.cuda()
    #     # model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = False
    #
    # model.eval()
    # model = model.cuda()
    # cudnn.benchmark = False
    # ------------------------------------------------------------------------------------------------------------------#

    total_imgs_bboxes_gt, total_imgs_path = load_test_dataset_iou(test_set, config)
    slice_idx = len(total_imgs_bboxes_gt) // gpu_count

    # last gpu
    if gpu_idx == gpu_count - 1:
        piece_imgs_path = total_imgs_path[gpu_idx * slice_idx:]
    else:
        piece_imgs_path = total_imgs_path[gpu_idx * slice_idx: (gpu_idx + 1) * slice_idx]

    # -----------------------------------------------------------------------------------------------------------------#
    # total_img_bboxes_pre = []
    for k, img_path in enumerate(piece_imgs_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(
            model,
            image,
            config.text_threshold,
            config.link_threshold,
            config.low_text,
            config.cuda,
            config.poly,
            config.canvas_size,
            config.mag_ratio,
        )

        # -------------------------------------------------------------------------------------------------------------#

        for box in bboxes:
            box_info = {"points": box, "text": "###", "ignore": False}
            single_img_bbox.append(box_info)
        # total_img_bboxes_pre.append(single_img_bbox)
        buffer[gpu_idx * slice_idx + k] = single_img_bbox
        # -------------------------------------------------------------------------------------------------------------#

        if config.vis_opt:
            viz_test(
                image,
                score_text,
                pre_box=polys,
                gt_box=total_imgs_bboxes_gt[k],
                img_name=img_path,
                result_dir=result_dir,
                test_folder_name=test_set,
            )

    # ------------------------------------------------------------------------------------------------------------------#

    # wait until buffer is full filled
    while None in buffer:
        continue
    assert all(v is not None for v in buffer), 'Buffer not filled'
    total_img_bboxes_pre = buffer

    # print('Predict bbox points completed.')
    results = []
    error_idx = []
    for i, (gt, pred) in enumerate(zip(total_imgs_bboxes_gt, total_img_bboxes_pre)):
        perSampleMetrics_dict = evaluator.evaluate_image(gt, pred)
        results.append(perSampleMetrics_dict)
        # if perSampleMetrics_dict["detCare"] != perSampleMetrics_dict["gtCare"]:
        #     error_idx.append(str(i))
    metrics = evaluator.combine_results(results)
    print(metrics)


    # save result
    with open(os.path.join(result_dir,"result.txt"), "w") as f:
        f.write(json.dumps(metrics))

    return metrics



# NOTE
def main_cleval(model_path, backbone, config, result_dir, model):

    # test 폴더에 대한 학습된 모델의 f1-score를 계산
    # test 폴더에 대한 model의 output 시각화
    # TODO loss 까지 구할 수 있도록?

    # model_path : 학습된 모델의 저장 경로
    # config : test에 필요한 configuration, dict type
    # evaluator : test function

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # print('Current cuda device:', torch.cuda.current_device())
    # print('Total gpu :', torch.cuda.device_count())
    gpu_idx = torch.cuda.current_device()
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)

    test_set = config.test_data_dir.split("/")[-2].lower()

    # load model
    # if backbone == "vgg":
    #     model = CRAFT()  # initialize
    # elif backbone == "resnet":
    #     model = UNetWithResnet50Encoder()
    # else:
    #     raise Exception('Undefined architecture')
    #
    # print("Loading weights from checkpoint (" + model_path + ")")
    # net_param = torch.load(model_path, map_location=f'cuda:{gpu_idx}')
    # try:
    #     model.load_state_dict(copyStateDict(net_param["craft"]))
    # except :
    #     model.load_state_dict(copyStateDict(net_param))
    #
    # if config.cuda:
    #     model = model.cuda()
    #     # model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = False

    model.eval()
    # model = model.cuda()
    # cudnn.benchmark = False
    # ------------------------------------------------------------------------------------------------------------------#

    total_imgs_bboxes_gt, total_imgs_path = load_test_dataset_cl(test_set, config)
    slice_idx = len(total_imgs_bboxes_gt) // gpu_count

    # last gpu
    if gpu_idx == gpu_count - 1:
        piece_imgs_path = total_imgs_path[gpu_idx * slice_idx:]
    else:
        piece_imgs_path = total_imgs_path[gpu_idx * slice_idx: (gpu_idx + 1) * slice_idx]

    # -----------------------------------------------------------------------------------------------------------------#
    # total_img_bboxes_pre = []
    for k, img_path in enumerate(piece_imgs_path):

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = img_path.split('/')[-1].split(".jpg")[0]
        bboxes, polys, score_text = test_net(
            model,
            image,
            config.text_threshold,
            config.link_threshold,
            config.low_text,
            config.cuda,
            config.poly,
            config.canvas_size,
            config.mag_ratio,
        )

        # -------------------------------------------------------------------------------------------------------------#

        if config.vis_opt:
            viz_test(
                image,
                score_text,
                pre_box=polys,
                gt_box=total_imgs_bboxes_gt[k],
                img_name=img_name,
                result_dir=result_dir,
                test_folder_name=test_set,
            )

        # -------------------------------------------------------------------------------------------------------------#

        result_pred = result_to_clEval(bboxes)
        make_txt(result_pred, result_dir, img_name, dtype='pred')

    # -----------------------------------------------------------------------------------------------------------------#

    gt_cl_dir = load_gt_cl_dir(config, data=test_set)

    if test_set == "icdar2013":
       GT_BOX_TYPE = "LTRB"
    else:
       GT_BOX_TYPE = "QUAD"

    metrics = clEval.main(gt_cl_dir, result_dir, GT_BOX_TYPE=GT_BOX_TYPE,PRED_BOX_TYPE="QUAD")

    print('Finish : cleval evaluation' + '-' * 50)

    # save result
    with open(os.path.join(result_dir,"result.txt"), "w") as f:
        f.write(json.dumps(metrics))

    return metrics

def cal_eval(config, data, res_dir_name, opt):
    evaluator = DetectionIoUEvaluator()
    # import ipdb; ipdb.set_trace()
    test_config = DotDict(config.test[data])
    res_dir = os.path.join(os.path.join("exp", args.yaml), "{}".format(res_dir_name))

    if opt == "iou_eval":
        main_eval(config.test.trained_model, config.train.backbone, test_config, evaluator, res_dir, buffer=None)
    elif opt == "cl_eval":
        main_cleval(config.test.trained_model, config.train.backbone, test_config, res_dir)
    else:
        print("not evaluation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detection Eval")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="syn_train_base_en_ko_ai",
        type=str,
        help="Load configuration",
    )
    args = parser.parse_args()

    # load configure
    config = load_yaml(args.yaml)
    config = DotDict(config)

    if config["wandb_opt"]:
        wandb.init(project="craft-stage1", entity="gmuffiness", name=args.yaml)
        # wandb.init(project="jm-test", entity="pingu", name=args.yaml)
        wandb.config.update(config)

    val_result_dir_name = args.yaml
    cal_eval(config, "icdar2013", val_result_dir_name + '-ic13-iou', opt="iou_eval")
    cal_eval(config, "icdar2013", val_result_dir_name + '-ic13-cl', opt="cl_eval")
    #cal_eval(config, "icdar2015", "resnet-en-ko-ai-15-iou", opt="iou_eval")
    cal_eval(config, "prescription", val_result_dir_name + '-pre-cl', opt="cl_eval")

