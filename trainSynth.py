import os
import cv2
import time
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
from craft import CRAFT
from utils import config
from loss.mseloss import Maploss, Maploss_v2, Maploss_v3
from data.dataset import SynthTextDataLoader
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import save_parser, make_logger, AverageMeter


parser = argparse.ArgumentParser(description='CRAFT SynthText Train')
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser.add_argument('--results_dir', default='./exp/synthtext/', type=str, help='Path to save checkpoints')
parser.add_argument('--synthData_dir', default='/data/SynthText/', type=str, help='Path to root directory of SynthText dataset')
parser.add_argument("--ckpt_path", default='', type=str, help="path to pretrained model")
parser.add_argument('--batch_size', default=16, type = int, help='batch size of training')
parser.add_argument('--st_iter', default=0, type = int, help='start iter')
parser.add_argument('--end_iter', default=10, type = int, help='end iter')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', default=10000, type=int, help='learning rate decay')
parser.add_argument('--gamma', '--gamma', default=0.8, type=float, help='initial gamma')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

parser.add_argument('--loss', default=3, type=int, help='loss version')
parser.add_argument('--neg_rto', default=3, type=int, help='negative pixel ratio')
parser.add_argument('--enlargeSize', default=0.75, type=float, help='enlargebox size')
parser.add_argument('--rnd_crop', default='rnd_back', type=str, help='random crop version')

parser.add_argument('--aug', action='store_true', help='augmentation')
parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision')

parser.add_argument('--wandb-name', default=None, type=str, help='name for wandb logging')

#for test
parser.add_argument('--trained_model', default='', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold') # ICDAR2015 0.85
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score') # ICDAR2015 0.5
parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold') # ICDAR2013: 0.2
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=960, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--isTraingDataset', default=False, type=str2bool, help='test for training or test data')
parser.add_argument('--test_folder', default='/data/ICDAR2013/', type=str, help='folder path to input images')

args = parser.parse_args()

#wandb.init(project="CRAFT", entity="pingu", name=args.wandb_name)
#wandb.config.update(args)

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

    if gpu == 0:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

    if gpu == 0:
        save_parser(args)
    config.AUG = args.aug
    config.ITER = args.st_iter

    batch_size = int(args.batch_size / ngpus_per_node)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:3457',
        world_size=ngpus_per_node,
        rank=gpu)


    synthData_dir = {"synthtext": args.synthData_dir}
    synthDataLoader = SynthTextDataLoader(args, target_size=768, data_dir_list=synthData_dir, mode='')
    #tst_charbox, tst_image, tst_imgtxt = synthDataLoader.load_synthtext(mode='test')
    #test_data_li = [tst_charbox, tst_image, tst_imgtxt]

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

    #logger
    if gpu == 0:
        trn_logger, val_logger = make_logger(path=args.results_dir)

    train_step = args.st_iter
    whole_training_step = args.end_iter
    update_lr_rate_step = 0
    training_lr = args.lr
    loss_value = 0
    batch_time = 0
    losses = AverageMeter()

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
            losses.update(loss.item(), images.size(0))

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
                val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])

                #wandb.log({"ICDAR2013 Recall": np.round(metrics['recall'], 3),
                #           "ICDAR2013 Precision": np.round(metrics['precision'], 3),
                #           "ICDAR2013 F1-score": np.round(metrics['hmean'], 3)})

                losses.reset()
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

        val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])

if __name__=='__main__':
    main()
