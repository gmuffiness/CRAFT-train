import os
import cv2
import time
import yaml
import shutil
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from eval import main as main_eval
from model.craft import CRAFT
from utils import config
from loss.mseloss import Maploss, Maploss_v2, Maploss_v3
from data.dataset import SynthTextDataLoader
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import save_parser
from config import config

parser = argparse.ArgumentParser(description='CRAFT SynthText Train')
parser.add_argument('--yaml_path', default='./exp/synthtext/', type=str, help='Load configuration')
parser.add_argument('--config_name', default='./exp/synthtext/', type=str, help='Load configuration')
args = parser.parse_args()



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

def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return param_group['lr']

def main():
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

def main_worker(gpu, ngpus_per_node):


    # ----------------------------------------------------------------------------------------------------------------#

    if gpu == 0:
        config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)

        # make result_dir
        res_dir_name = args.yaml_path.split('/')[-1].split(".yaml")[0]
        res_dir = os.path.join('exp', res_dir_name)
        config["results_dir"] = res_dir

        if not os.path.exists(res_dir): os.makedirs(res_dir)
        # Duplicate yaml file to result_dir
        shutil.copy(args.yaml_path, os.path.join(res_dir, res_dir_name) + '.yaml')

        # Apply config to wandb
        wandb.init(project="CRAFT", entity="pingu", name=res_dir_name)
        wandb.config.update(config)

    # ----------------------------------------------------------------------------------------------------------------#


    config.AUG = args.aug
    config.ITER = args.st_iter
    batch_size = int(args.batch_size / ngpus_per_node)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:3455',
        world_size=ngpus_per_node,
        rank=gpu)



    import ipdb;ipdb.set_trace()
    synthData_dir = {"synthtext": args.synthData_dir}
    synthDataLoader = SynthTextDataLoader(target_size=config.train.data.output_size, data_dir=config.data_dir.synthtext, logging=config.train.data.logging)


    train_sampler = torch.utils.data.distributed.DistributedSampler(synthDataLoader)
    train_loader = torch.utils.data.DataLoader(synthDataLoader,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               sampler=train_sampler,
                                               drop_last=False,
                                               pin_memory=True)


    craft = CRAFT(pretrained=True, amp=args.amp)
    craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft)
    torch.cuda.set_device(gpu)
    craft = craft.cuda(gpu)
    craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[gpu])

    if args.st_iter != 0:
        print('success craft_load')
        net_param = torch.load(args.ckpt_path)
        try:
            craft.load_state_dict(copyStateDict(net_param['craft']))
        except:
            craft.load_state_dict(copyStateDict(net_param))


    # craft = torch.nn.DataParallel(craft).cuda()
    torch.backends.cudnn.benchmark =True

    optimizer = optim.Adam(craft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.st_iter != 0:
        print('success optim_load')
        optimizer.load_state_dict(copyStateDict(net_param['optimizer']))
        args.st_iter = net_param['optimizer']['state'][0]['step']
        args.lr = net_param['optimizer']['param_groups'][0]['lr']

    # mixed precision
    if args.amp :
        if args.st_iter != 0 :
            scaler = torch.cuda.amp.GradScaler()
            scaler.load_state_dict(copyStateDict(net_param["scaler"]))
        else:
            scaler = torch.cuda.amp.GradScaler()

    if args.loss == 2:
        criterion = Maploss_v2()
    elif args.loss == 3:
        criterion = Maploss_v3()



    train_step = args.st_iter
    whole_training_step = args.end_iter
    update_lr_rate_step = 0
    training_lr = args.lr
    loss_value = 0
    batch_time = 0


    start_time = time.time()
    while train_step < whole_training_step:

        for index, (image, region_image, affinity_image, confidence_mask, confidences) in enumerate(train_loader):
            craft.train()
            if train_step>0 and train_step % args.lr_decay==0:
                update_lr_rate_step += 1
                training_lr = adjust_learning_rate(optimizer, args.gamma, update_lr_rate_step, args.lr)

            images = Variable(image).cuda()
            region_image_label = Variable(region_image).cuda()
            affinity_image_label = Variable(affinity_image).cuda()
            confidence_mask_label = Variable(confidence_mask).cuda()

            if args.amp:
                with torch.cuda.amp.autocast():
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label, args.neg_rto)
            else:
                output, _ = craft(images)
                out1 = output[:, :, :, 0]
                out2 = output[:, :, :, 1]
                loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label, args.neg_rto)

            optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            end_time = time.time()
            loss_value += loss.item()
            batch_time += (end_time - start_time)


            #wandb.log({"SynthText Loss": loss.item()})

            if train_step % 50000 == 0 and gpu == 0: # and train_step != 0

                print('Saving state, index:', train_step)

                if args.amp:
                    torch.save({
                        'iter': train_step,
                        'craft': craft.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "scaler": scaler.state_dict()
                    }, args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth')

                else:
                    torch.save({
                        'iter': train_step,
                        'craft': craft.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth')


                evaluator = DetectionIoUEvaluator()
                if args.amp:
                    metrics = main_eval(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args, evaluator)
                else:
                    metrics = main_eval(args.results_dir + '/CRAFT_clr_' + repr(train_step)+ '.pth', args, evaluator)


                #wandb.log({"ICDAR2013 Recall": np.round(metrics['recall'], 3),
                #           "ICDAR2013 Precision": np.round(metrics['precision'], 3),
                #           "ICDAR2013 F1-score": np.round(metrics['hmean'], 3)})


            train_step += 1
            config.ITER +=1

            if train_step >= whole_training_step : break

    if gpu == 0:
        if args.amp:
            torch.save({
                'iter': train_step,
                'craft': craft.state_dict(),
                'optimizer': optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }, args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth')


        else:
            torch.save({
                'iter': train_step,
                'craft': craft.state_dict(),
                'optimizer': optimizer.state_dict()
            }, args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth')


        evaluator = DetectionIoUEvaluator()
        if args.amp:
            metrics = main_eval(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args, evaluator)
        else:
            metrics = main_eval(args.results_dir + '/CRAFT_clr_' + repr(train_step)+ '.pth', args, evaluator)



if __name__=='__main__':
    main()
