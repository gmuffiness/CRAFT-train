# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, ICDAR2015
from eval import main as main_eval
from loss.mseloss import Maploss, Maploss_v2, Maploss_v3
from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser
import utils.config as temp_config
from utils.decorator_wraps import time_printer

class Trainer(object):
    def __init__(self, config, gpu):

        self.config = config
        self.gpu = gpu
        self.net_param = self.get_load_param(gpu)

    def get_synth_loader(self):

        synth_dataset = SynthTextDataSet(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_dir.synthtext,
            saved_gt_dir=self.config.data_dir.synthtext_gt,
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
        )

        synth_sampler = torch.utils.data.distributed.DistributedSampler(synth_dataset)
        synth_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=self.config.train.batch_size//5,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            sampler=synth_sampler,
            drop_last=False,
            pin_memory=True,
        )

        return synth_loader

    def get_icdar_dataset(self):

        icdar15_dataset = ICDAR2015(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_dir.ic15,
            saved_gt_dir=self.config.data_dir.ic15_gt,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.icdar_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.icdar_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        return icdar15_dataset

    def get_load_param(self, gpu):

        if self.config.train.ckpt_path is not None:
            # map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            map_location = 'cuda:%d' % gpu
            param = torch.load(self.config.train.ckpt_path, map_location=map_location)
        else:
            param = None

        return param

    def adjust_learning_rate(self, optimizer, gamma, step, lr):
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

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def train(self):

        torch.cuda.set_device(self.gpu)
        total_gpu_num = torch.cuda.device_count()

        # MODEL -------------------------------------------------------------------------------------------------------#
        # SUPERVISION model
        if self.config.data_dir.ic15_gt is None:
            supervision_model = CRAFT(pretrained=True, amp=self.config.train.amp)
            # Only useful on half GPU train / half GPU supervision setting
            supervision_device = total_gpu_num // 2 + self.gpu
            if self.config.train.ckpt_path is not None:
                # supervision_model.load_state_dict(copyStateDict(self.net_param['craft']))
                supervision_param = self.get_load_param(supervision_device)
                supervision_model.load_state_dict(copyStateDict(supervision_param['craft']))
                supervision_model = supervision_model.to(f'cuda:{supervision_device}')
            print(f'Supervision model loading on : gpu {supervision_device}')
        else:
            supervision_model, supervision_device = None, None

        # TRAIN model
        craft = CRAFT(pretrained=True, amp=self.config.train.amp)
        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param['craft']))

        craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft)
        craft = craft.cuda()
        craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[self.gpu])

        torch.backends.cudnn.benchmark = True

        # DATASET -----------------------------------------------------------------------------------------------------#
        trn_syn_loader = self.get_synth_loader()
        batch_syn = iter(trn_syn_loader)
        trn_icdar_dataset = self.get_icdar_dataset()
        if self.config.data_dir.ic15_gt is None:
            trn_icdar_dataset.update_model(supervision_model)
            trn_icdar_dataset.update_device(supervision_device)

        trn_icdar15_sampler = torch.utils.data.distributed.DistributedSampler(trn_icdar_dataset)
        trn_icdar_loader = torch.utils.data.DataLoader(
            trn_icdar_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            sampler=trn_icdar15_sampler,
            drop_last=False,
            pin_memory=True,
        )

        # OPTIMIZER ---------------------------------------------------------------------------------------------------#
        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

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
        start_time = time.time()

        print("================================ Train start ================================")
        while train_step < whole_training_step:
            trn_icdar15_sampler.set_epoch(train_step)
            for index, (
                icdar_image,
                icdar_region_label,
                icdar_affi_label,
                icdar_confidence_mask,
            ) in enumerate(trn_icdar_loader):
                craft.train()
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                # syn image load
                syn_image, syn_region_label, syn_affi_label, syn_confidence_mask = next(batch_syn)

                # load data to each GPU
                syn_image = syn_image.cuda(non_blocking=True)
                icdar_image = icdar_image.cuda(non_blocking=True)
                syn_region_label = syn_region_label.cuda(non_blocking=True)
                icdar_region_label = icdar_region_label.cuda(non_blocking=True)
                syn_affi_label = syn_affi_label.cuda(non_blocking=True)
                icdar_affi_label = icdar_affi_label.cuda(non_blocking=True)
                syn_confidence_mask = syn_confidence_mask.cuda(non_blocking=True)
                icdar_confidence_mask = icdar_confidence_mask.cuda(non_blocking=True)

                # cat syn & icdar image
                images = torch.cat((syn_image, icdar_image), 0)
                region_image_label = torch.cat((syn_region_label, icdar_region_label), 0)
                affinity_image_label = torch.cat((syn_affi_label, icdar_affi_label), 0)
                confidence_mask_label = torch.cat((syn_confidence_mask, icdar_confidence_mask), 0)

                # only use ic15 data (for fast debugging)
                # images = Variable(icdar_image).cuda()
                # region_image_label = Variable(icdar_region_label.type(torch.FloatTensor)).cuda()
                # affinity_image_label = Variable(icdar_affi_label.type(torch.FloatTensor)).cuda()
                # confidence_mask_label = Variable(icdar_confidence_mask.type(torch.FloatTensor)).cuda()

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

                    if self.config.wandb_opt:
                        wandb.log({'train_step': train_step, 'mean_loss': mean_loss})


                if train_step % self.config.train.eval_interval == 0 and train_step != 0 and self.gpu == 0:

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
                    #
                    if self.config.wandb_opt:
                        wandb.log(
                            {
                                "ICDAR2015 Recall": np.round(metrics["recall"], 3),
                                "ICDAR2015 Precision": np.round(metrics["precision"], 3),
                                "ICDAR2015 F1-score": np.round(metrics["hmean"], 3),
                            }
                        )

                train_step += 1
                temp_config.ITER = train_step
                if train_step >= whole_training_step:
                    break
            state_dict = craft.module.state_dict()
            supervision_model.load_state_dict(state_dict)
            trn_icdar_dataset.update_model(supervision_model)

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

            evaluator = DetectionIoUEvaluator()
            val_result_dir = os.path.join(
                self.config.results_dir, "{}".format(str(train_step))
            )
            metrics = main_eval(save_param_path, self.config, evaluator, val_result_dir)

            if self.config.wandb_opt:
                wandb.log(
                    {
                        "ICDAR2015 Recall": np.round(metrics["recall"], 3),
                        "ICDAR2015 Precision": np.round(metrics["precision"], 3),
                        "ICDAR2015 F1-score": np.round(metrics["hmean"], 3),
                    }
                )
                wandb.finish()


def main():

    # Start train
    ngpus_per_node = torch.cuda.device_count() // 2
    # ngpus_per_node = 4
    world_size = ngpus_per_node

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

def main_worker(gpu, ngpus_per_node):
    parser = argparse.ArgumentParser(description="CRAFT IC15 Train")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="ic15_test6_26",
                        type=str,
                        help="Load configuration")

    parser.add_argument("--port",
                        "--use ddp port",
                        default="2346",
                        type=str,
                        help="Port number")

    args = parser.parse_args()


    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + args.port,
        world_size=ngpus_per_node,
        rank=gpu)



    # load configure
    config = load_yaml(args.yaml)

    if gpu == 0:
        # Apply config to wandb
        if config["wandb_opt"]:
            wandb.init(project="craft-icdar", entity="gmuffiness", name=args.yaml)
            wandb.config.update(config)
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

    batch_size = int(config["train"]["batch_size"] / ngpus_per_node)
    config["train"]["batch_size"] = batch_size
    config = DotDict(config)

    # Start train
    trainer = Trainer(config, gpu)
    trainer.train()


if __name__ == "__main__":
    main()
