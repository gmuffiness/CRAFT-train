import os
import time
import wandb
import argparse
from collections import OrderedDict
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import shutil

from eval import main as main_eval
from model.craft import CRAFT
from loss.mseloss import Maploss_v2, Maploss_v3
from data.dataset import SynthTextDataLoader
from metrics.eval_det_iou import DetectionIoUEvaluator


class Trainer(object):
    def __init__(self, config):

        self.config = config
        self.trn_config = config["train"]
        self.synth_loader = self._get_synth_loader()
        self.net_param = self._get_load_param()



    def _copy_state_dict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def _adjust_learning_rate(self, optimizer, gamma, step, lr):
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

    def _get_synth_loader(self):
        # 나중에 따로 동작할 수 도 있을 것 같아서 분리 시켜 놓음

        synthDataLoader = SynthTextDataLoader(self.config)
        synth_sampler = torch.utils.data.distributed.DistributedSampler(synthDataLoader)
        synth_loader = torch.utils.data.DataLoader(synthDataLoader,
                                                   batch_size=self.trn_config["batch_size"],
                                                   shuffle=False,
                                                   num_workers=self.trn_config["num_workers"],
                                                   sampler=synth_sampler,
                                                   drop_last=False,
                                                   pin_memory=True)

        return synth_loader

    def _get_load_param(self):

        if self.trn_config["ckpt_path"] is not None:
            param = torch.load(self.trn_config["ckpt_path"])
        else:
            param = None

        return param


    def _get_loss(self):
        if self.trn_config["loss"] == 2:
            criterion = Maploss_v2()
        elif self.trn_config["loss"] == 3:
            criterion = Maploss_v3()
        return criterion



    def train(self, gpu):

        trn_loader = self.synth_loader
        # -------------------------------------------------------------------------------------------------------#

        craft = CRAFT(pretrained=True, amp=self.trn_config["amp"])
        craft = nn.SyncBatchNorm.convert_sync_batchnorm(craft)
        torch.cuda.set_device(gpu)
        craft = craft.cuda(gpu)
        craft = torch.nn.parallel.DistributedDataParallel(craft, device_ids=[gpu])


        # load model
        if self.trn_config["ckpt_path"] is not None:
            craft.load_state_dict(self.copy_state_dict(self.net_param["train"]['craft']))
            print('success craft_load.')


        torch.backends.cudnn.benchmark = True

        # ----------------------------------------------------------------------------------------------------------#

        optimizer = optim.Adam(craft.parameters(), lr=self.trn_config["lr"],
                               weight_decay=self.trn_config["weight_decay"])

        # load optim
        if self.trn_config["ckpt_path"] is not None:
            optimizer.load_state_dict(self.copy_state_dict(self.net_param['optimizer']))
            self.trn_config["st_iter"] = self.net_param['optimizer']['state'][0]['step']
            self.trn_config["lr"] = self.net_param['optimizer']['param_groups'][0]['lr']
            print('success optim_load')

        # ---------------------------------------------------------------------------------------------------------#

        # mixed precision
        if self.trn_config["amp"]:
            scaler = torch.cuda.amp.GradScaler()

            if self.trn_config["ckpt_path"] is not None:
                scaler.load_state_dict(self.copy_state_dict(self.net_param["scaler"]))

        # loss
        criterion = self._get_loss()

        # ------------------------------------------------------------------------------------------------------#

        train_step = self.trn_config["st_iter"]
        whole_training_step = self.trn_config["end_iter"]
        update_lr_rate_step = 0
        training_lr = self.trn_config["lr"]
        loss_value = 0
        batch_time = 0

        start_time = time.time()
        while train_step < whole_training_step:
            for index, (image, region_image, affinity_image, confidence_mask) in enumerate(
                    trn_loader):
                craft.train()
                if train_step > 0 and train_step % self.trn_config["lr_decay"] == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(optimizer, self.trn_config["gamma"],
                                                       update_lr_rate_step, self.trn_config["lr"])

                images = Variable(image).cuda()
                region_image_label = Variable(region_image).cuda()
                affinity_image_label = Variable(affinity_image).cuda()
                confidence_mask_label = Variable(confidence_mask).cuda()

                if self.trn_config["amp"]:
                    with torch.cuda.amp.autocast():
                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]
                        loss = criterion(region_image_label, affinity_image_label,
                                         out1, out2, confidence_mask_label, self.trn_config["neg_rto"])

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()


                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(region_image_label, affinity_image_label,
                                     out1, out2, confidence_mask_label, self.trn_config["neg_rto"])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                end_time = time.time()
                loss_value += loss.item()
                batch_time += (end_time - start_time)
                if gpu == 0:
                    wandb.log({"SynthText Loss": loss.item()})


                if train_step % 50000 == 0 and train_step != 0:

                    print('Saving state, index:', train_step)
                    save_param_dic = {'iter': train_step,
                                      'craft': craft.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      }
                    save_param_path = self.config["results_dir"] + '/CRAFT_clr_' + repr(train_step) + '.pth'

                    if self.trn_config["amp"]:
                        save_param_dic["scaler"] =  scaler.state_dict()
                        save_param_path = self.config["results_dir"] + '/CRAFT_clr_amp_' + repr(train_step) + '.pth'

                    torch.save(save_param_dic,save_param_path)

                    # validation
                    evaluator = DetectionIoUEvaluator()
                    metrics = main_eval(save_param_path, self.config, evaluator)


                    # wandb.log({"ICDAR2013 Recall": np.round(metrics['recall'], 3),
                    #           "ICDAR2013 Precision": np.round(metrics['precision'], 3),
                    #           "ICDAR2013 F1-score": np.round(metrics['hmean'], 3)})

                train_step += 1
                if train_step >= whole_training_step: break



        # save last model
        save_param_dic = {'iter': train_step,
                          'craft': craft.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          }
        save_param_path = self.config["results_dir"] + '/CRAFT_clr_' + repr(train_step) + '.pth'

        if self.trn_config["amp"]:
            save_param_dic["scaler"] = scaler.state_dict()
            save_param_path = self.config["results_dir"] + '/CRAFT_clr_amp_' + repr(train_step) + '.pth'
        torch.save(save_param_dic, save_param_path)

        evaluator = DetectionIoUEvaluator()
        metrics = main_eval(save_param_path, self.config, evaluator)

        # wandb.log({"ICDAR2013 Recall": np.round(metrics['recall'], 3),
        #           "ICDAR2013 Precision": np.round(metrics['precision'], 3),
        #           "ICDAR2013 F1-score": np.round(metrics['hmean'], 3)})



def main():

    # Start train
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

def main_worker(gpu, ngpus_per_node):

    parser = argparse.ArgumentParser(description='CRAFT SynthText Train')
    parser.add_argument('--yaml_path', default='./exp/synthtext/', type=str, help='Load configuration')
    args = parser.parse_args()


    config = yaml.load(open(args.yaml_path, "r"), Loader=yaml.FullLoader)

    # Make result_dir
    res_dir_name = args.yaml_path.split('/')[-1].split(".yaml")[0]
    res_dir = os.path.join('exp', res_dir_name)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    # batch size
    config["train"]["batch_size"] = int(config["train"]["batch_size"] / ngpus_per_node)

    # Duplicate yaml file to result_dir
    shutil.copy(args.yaml_path, os.path.join(res_dir, res_dir_name) + '.yaml')



    if gpu == 0:
        # Apply config to wandb
        wandb.init(project="jm-test", entity="pingu", name=res_dir_name)
        wandb.config.update(config)


    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:3455',
        world_size=ngpus_per_node,
        rank=gpu)



    trainer = Trainer(config)
    trainer.train(gpu)


if __name__=='__main__':
    main()
