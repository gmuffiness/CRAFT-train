# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import yaml
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import wandb

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, SynthTextDataSet_KR, hierarchical_dataset, AiHubDataset
from eval_v2 import main_eval, main_cleval
from loss.mseloss import Maploss, Maploss_v2, Maploss_v3
from model.craft import CRAFT
from model.craft_resnet import UNetWithResnet50Encoder
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser


class Trainer(object):
    def __init__(self, config, gpu):

        self.config = config
        self.gpu = gpu
        self.trn_loader, self.trn_sampler = self.get_trn_loader()
        self.net_param = self.get_load_param(gpu)

    def get_trn_loader(self):

        total_trn_dataset = []

        if "synthtext" in self.config.train.dataset:
            # eng-syn
            synth_dataset = SynthTextDataSet(
                output_size=self.config.train.data.output_size,
                data_dir=self.config.data_dir.synthtext,
                saved_gt_dir=None,
                mean=self.config.train.data.mean,
                variance=self.config.train.data.variance,
                gauss_init_size=self.config.train.data.gauss_init_size,
                gauss_sigma=self.config.train.data.gauss_sigma,
                enlarge_region=self.config.train.data.enlarge_region,
                enlarge_affinity=self.config.train.data.enlarge_affinity,
                aug=self.config.train.data.syn_aug,
                vis_test_dir=self.config.vis_test_dir,
                vis_opt=self.config.train.data.vis_opt,
                sample=self.config.train.data.syn_sample
            )
            total_trn_dataset.append(synth_dataset)

        if "ai_hub" in self.config.train.dataset:
            ai_hub_dataset = AiHubDataset(
                output_size=self.config.train.data.output_size,
                data_dir=self.config.data_dir.ai_hub,
                saved_gt_dir=None,
                mean=self.config.train.data.mean,
                variance=self.config.train.data.variance,
                gauss_init_size=self.config.train.data.gauss_init_size,
                gauss_sigma=self.config.train.data.gauss_sigma,
                enlarge_region=self.config.train.data.enlarge_region,
                enlarge_affinity=self.config.train.data.enlarge_affinity,
                aug=self.config.train.data.syn_aug,
                vis_test_dir=self.config.vis_test_dir,
                vis_opt=self.config.train.data.vis_opt,
                sample=self.config.train.data.syn_sample,
                do_not_care_label=self.config.train.data.do_not_care_label,
            )

            total_trn_dataset.append(ai_hub_dataset)

        if "synthtext_kor" in self.config.train.dataset:
            # kor-syn
            data_path_kr = self.config.data_dir.synthtext_kor
            total_trn_dataset.extend(hierarchical_dataset(root=data_path_kr, config=self.config))

        total_trn_dataset = ConcatDataset(total_trn_dataset)

        trn_sampler = torch.utils.data.distributed.DistributedSampler(total_trn_dataset)

        # mp_context = torch.multiprocessing.get_context('fork')
        trn_loader = torch.utils.data.DataLoader(
            total_trn_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            sampler=trn_sampler,
            drop_last=True,
            pin_memory=True,
            # multiprocessing_context=mp_context,
        )

        return trn_loader, trn_sampler

    def get_load_param(self, gpu):

        if self.config.train.ckpt_path is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            param = torch.load(self.config.train.ckpt_path, map_location=map_location)
        else:
            param = None

        return param


    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        """Sets the learning rate to the initial LR decayed by gamma value at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = lr * (gamma**step)
        print(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset, train_step, save_param_path, buffer, model):
        # Input dataset : "icdar2013" |  "icdar2015" | "prescription"

        test_config = DotDict(self.config.test[dataset])

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset+"_iou", str(train_step))
        )

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            save_param_path, self.config.train.backbone, test_config, evaluator, val_result_dir, buffer, model
        )
        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} IoU Recall".format(dataset): np.round(metrics["recall"], 3),
                    "{} IoU Precision".format(dataset): np.round(metrics["precision"], 3),
                    "{} IoU F1-score".format(dataset): np.round(metrics["hmean"], 3),
                }
            )

    def cleval(self, dataset, train_step, save_param_path, model):
        # Input dataset : "icdar2013" |  "icdar2015" | "prescription"

        test_config = DotDict(self.config.test[dataset])

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset+"_cl", str(train_step))
        )

        metrics = main_cleval(
            save_param_path, self.config.train.backbone, test_config, val_result_dir, model
        )

        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} CLeval Recall".format(dataset): np.round(metrics["recall"], 3),
                    "{} CLeval Precision".format(dataset): np.round(metrics["precision"], 3),
                    "{} CLeval F1-score".format(dataset): np.round(metrics["hmean"], 3),
                }
            )

    def train(self, buffer_dict):

        torch.cuda.set_device(self.gpu)

        # DATASET -----------------------------------------------------------------------------------------------------#
        trn_loader = self.trn_loader

        # MODEL -------------------------------------------------------------------------------------------------------#
        if self.config.train.backbone == "vgg":
            craft = CRAFT(pretrained=True, amp=self.config.train.amp)
        elif self.config.train.backbone == "resnet":
            craft = UNetWithResnet50Encoder(pretrained=True, amp=self.config.train.amp)
        else:
            raise Exception('Undefined architecture')

        # load model
        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param["craft"]))
        craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft)
        craft = craft.cuda()
        craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[self.gpu])

        torch.backends.cudnn.benchmark = True

        # OPTIMIZER----------------------------------------------------------------------------------------------------#

        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        # load optim
        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

            # load model
            if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
                scaler.load_state_dict(copyStateDict(self.net_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN -------------------------------------------------------------------------------------------------------#
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0
        epoch = 0
        start_time = time.time()

        while train_step < whole_training_step:
            self.trn_sampler.set_epoch(train_step)
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

                # load data to each GPU async
                images = image.cuda(non_blocking=True)
                region_image_label = region_image.cuda(non_blocking=True)
                affinity_image_label = affinity_image.cuda(non_blocking=True)
                confidence_mask_label = confidence_mask.cuda(non_blocking=True)

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
                            self.config.train.n_min_neg
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

                if train_step > 0 and train_step%5==0 and self.gpu == 0:
                    mean_loss = loss_value / 5
                    loss_value = 0
                    avg_batch_time = batch_time/5
                    batch_time = 0

                    print("{}, training_step: {}|{}, learning rate: {:.8f}, "
                          "training_loss: {:.5f}, avg_batch_time: {:.5f}"
                          .format(time.strftime('%Y-%m-%d:%H:%M:%S',time.localtime(time.time())),
                                  train_step, whole_training_step, training_lr, mean_loss, avg_batch_time))
                    if self.gpu == 0 and self.config.wandb_opt:
                        wandb.log({'train_step': train_step, 'mean_loss': mean_loss})

                if train_step % self.config.train.eval_interval == 0 and train_step != 0:
                    craft.eval()
                    # initialize all buffer with zero
                    if self.gpu == 0:
                        for buffer in buffer_dict.values():
                            for i in range(len(buffer)):
                                buffer[i] = None

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
                    self.iou_eval("icdar2013", train_step, save_param_path, buffer_dict["icdar2013"], craft)
                    self.cleval("icdar2013", train_step, save_param_path, craft)
                    self.cleval("prescription", train_step, save_param_path, craft)
                    # self.iou_eval("prescription", train_step, save_param_path, buffer_dict["prescription"])
                    # self.cleval("icdar2015", train_step, save_param_path)

                train_step += 1
                if train_step >= whole_training_step:
                    break
            epoch += 1

        # save last model
        if self.gpu == 0:
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
            # NOTE
            self.iou_eval("icdar2013", train_step, save_param_path)
            #self.cleval("icdar2013", train_step, save_param_path)
            self.cleval("prescription", train_step, save_param_path)

            if self.config.wandb_opt:
                wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="CRAFT SynthText Train")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="./exp/synthtext/",
                        type=str,
                        help="Load configuration")

    parser.add_argument("--port",
                        "--use ddp port",
                        default="2646",
                        type=str,
                        help="Load configuration")

    args = parser.parse_args()

    exp_name = args.yaml
    # load configure
    config = load_yaml(args.yaml)

    print("-"*20+" Options "+"-"*20)
    print(yaml.dump(config))
    print("-" * 40)

    # Make result_dir
    res_dir = os.path.join("exp", args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(
        "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
    )


    ngpus_per_node = torch.cuda.device_count()
    print(f'Total device num : {ngpus_per_node}')
    world_size = ngpus_per_node

    manager = mp.Manager()
    buffer1 = manager.list([None] * config["test"]["icdar2013"]["test_set_size"])
    buffer2 = manager.list([None] * config["test"]["icdar2015"]["test_set_size"])
    buffer3 = manager.list([None] * config["test"]["prescription"]["test_set_size"])
    buffer_dict = {"icdar2013":buffer1, "icdar2015":buffer2, "prescription":buffer3}
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(args.port, ngpus_per_node, config, buffer_dict, exp_name, ))


def main_worker(gpu, port, ngpus_per_node, config, buffer_dict, exp_name):

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + port,
        world_size=ngpus_per_node,
        rank=gpu)

    if gpu == 0 and config["wandb_opt"]:
        # Apply config to wandb
        # wandb.init(project="jm-test", entity="pingu", name=args.yaml)
        wandb.init(project="craft-stage1", entity="gmuffiness", name=exp_name)
        wandb.config.update(config)

    batch_size = int(config["train"]["batch_size"] / ngpus_per_node)
    config["train"]["batch_size"] = batch_size
    config = DotDict(config)

    # Start train
    trainer = Trainer(config, gpu)
    trainer.train(buffer_dict)


if __name__ == "__main__":
    main()