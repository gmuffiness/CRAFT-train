# -*- coding: utf-8 -*-
import argparse
from collections import OrderedDict
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet
from eval import main as main_eval
from loss.mseloss import Maploss, Maploss_v2, Maploss_v3
from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser


class Trainer(object):
    def __init__(self, config):

        self.config = config
        self.synth_loader = self._get_synth_loader()
        self.net_param = self._get_load_param()

    def _get_synth_loader(self):
        # 나중에 따로 동작할 수 도 있을 것 같아서 분리 시켜 놓음

        synth_dataset = SynthTextDataSet(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_dir.synthtext,
            saved_gt_dir=self.config.data_dir.synthtext_gt,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_size=self.config.train.data.enlarge_size,
            aug=self.config.train.data.aug,
            logging=self.config.train.data.logging,
        )

        synth_sampler = torch.utils.data.distributed.DistributedSampler(synth_dataset)
        synth_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            sampler=synth_sampler,
            drop_last=False,
            pin_memory=True,
        )

        return synth_loader

    def _get_load_param(self):

        if self.config.train.ckpt_path is not None:
            param = torch.load(self.config.ckpt_path)
        else:
            param = None

        return param

    def _adjust_learning_rate(self, optimizer, gamma, step, lr):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = lr * (gamma**step)
        print(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def _get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        return criterion

    def train(self, gpu):
        if gpu == 0:
            print("start training")

        trn_loader = self.synth_loader
        # -------------------------------------------------------------------------------------------------------#
        craft = CRAFT(pretrained=True, amp=self.config.train.amp)
        craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft)
        torch.cuda.set_device(gpu)
        craft = craft.cuda(gpu)
        craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[gpu])

        # load model
        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param["craft"]))

        torch.backends.cudnn.benchmark = True
        # ----------------------------------------------------------------------------------------------------------#

        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        # load optim
        if self.config.train.ckpt_path is not None:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]


        # ---------------------------------------------------------------------------------------------------------#

        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

            # load model
            if self.config.train.ckpt_path is not None:
                craft.load_state_dict(copyStateDict(self.net_param["scaler"]))

        # loss
        criterion = self._get_loss()

        # ------------------------------------------------------------------------------------------------------#

        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0

        start_time = time.time()
        while train_step < whole_training_step:
            for index, (
                image,
                region_image,
                affinity_image,
                confidence_mask,
            ) in enumerate(trn_loader):

                craft.train()
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                images = Variable(image).cuda()
                region_image_label = Variable(region_image).cuda()
                affinity_image_label = Variable(affinity_image).cuda()
                confidence_mask_label = Variable(confidence_mask).cuda()

                if self.config.train.amp:
                    with torch.cuda.amp.autocast():

                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]
                        loss = criterion(
                            region_image_label,
                            affinity_image_label,
                            out1,
                            out2,
                            confidence_mask_label,
                            self.config.train.neg_rto,
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_image_label,
                        affinity_image_label,
                        out1,
                        out2,
                        confidence_mask_label,
                        self.config.train.neg_rto,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                loss_value += loss.item()
                batch_time += end_time - start_time

                # if gpu == 0:
                #     wandb.log({"SynthText Loss": loss.item()})

                if train_step % 50 == 0 and train_step != 0 and gpu == 0:

                    print("Saving state, index:", train_step)
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = (
                        self.config.results_dir
                        + "/CRAFT_clr_"
                        + repr(train_step)
                        + ".pth"
                    )

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()
                        save_param_path = (
                            self.config.results_dir
                            + "/CRAFT_clr_amp_"
                            + repr(train_step)
                            + ".pth"
                        )

                    torch.save(save_param_dic, save_param_path)

                    # validation
                    evaluator = DetectionIoUEvaluator()
                    val_result_dir = os.path.join(
                        self.config.results_dir, "{}".format(str(train_step))
                    )
                    metrics = main_eval(
                        save_param_path, self.config, evaluator, val_result_dir
                    )

                    # wandb.log(
                    #     {
                    #         "ICDAR2013 Recall": np.round(metrics["recall"], 3),
                    #         "ICDAR2013 Precision": np.round(metrics["precision"], 3),
                    #         "ICDAR2013 F1-score": np.round(metrics["hmean"], 3),
                    #     }
                    # )

                train_step += 1
                if train_step >= whole_training_step:
                    break

        # save last model
        if gpu == 0:
            save_param_dic = {
                "iter": train_step,
                "craft": craft.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_param_path = (
                self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"
            )

            if self.config.train.amp:
                save_param_dic["scaler"] = scaler.state_dict()
                save_param_path = (
                    self.config.results_dir + "/CRAFT_clr_amp_" + repr(train_step) + ".pth"
                )
            torch.save(save_param_dic, save_param_path)

            evaluator = DetectionIoUEvaluator()
            val_result_dir = os.path.join(
                self.config.results_dir, "{}".format(str(train_step))
            )
            metrics = main_eval(save_param_path, self.config, evaluator, val_result_dir)

            # wandb.log(
            #     {
            #         "ICDAR2013 Recall": np.round(metrics["recall"], 3),
            #         "ICDAR2013 Precision": np.round(metrics["precision"], 3),
            #         "ICDAR2013 F1-score": np.round(metrics["hmean"], 3),
            #     }
            # )
            wandb.finish()


def main():

    # Start train
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))


def main_worker(gpu, ngpus_per_node):

    parser = argparse.ArgumentParser(description="CRAFT SynthText Train")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="./exp/synthtext/",
                        type=str,
                        help="Load configuration")

    parser.add_argument("--port",
                        "--use ddp port",
                        default="2346",
                        type=str,
                        help="Load configuration")

    args = parser.parse_args()


    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + args.port,
        world_size=ngpus_per_node,
        rank=gpu)



    # load configure
    config = load_yaml(args.yaml)

    if gpu == 0:
        # Make result_dir
        res_dir = os.path.join("exp", args.yaml)
        config["results_dir"] = res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Duplicate yaml file to result_dir
        shutil.copy(
            "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
        )

        # Apply config to wandb
        wandb.init(project="jm-test", entity="pingu", name=args.yaml)
        wandb.config.update(config)


    batch_size = int(config["train"]["batch_size"] / ngpus_per_node)
    config["train"]["batch_size"] = batch_size
    config = DotDict(config)

    # Start train
    trainer = Trainer(config)
    trainer.train(gpu)


if __name__ == "__main__":
    main()
