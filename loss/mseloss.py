import random

import numpy as np
import torch
import torch.nn as nn
import cv2

import wandb

class Maploss(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss, self).__init__()

    def single_image_loss(self, pre_loss, loss_label):

        batch_size = pre_loss.shape[0]
        # sum_loss = torch.mean(pre_loss.view(-1))*0
        # pre_loss = pre_loss.view(batch_size, -1)
        # loss_label = loss_label.view(batch_size, -1)

        positive_pixel = (loss_label > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pre_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        negative_pixel = (loss_label <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)

        if negative_pixel_number < 3 * positive_pixel_number:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
        else:
            negative_loss_region = pre_loss * negative_pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region.view(-1),
                                                 int(3 * positive_pixel_number))[0]) / (positive_pixel_number * 3)

        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(self, region_scores_label, affinity_socres_label, region_scores_pre, affinity_scores_pre, mask):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_socres_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        # loss1 = torch.sqrt(loss1 + 1e-8)
        # loss2 = torch.sqrt(loss2 + 1e-8)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_region, region_scores_label)
        affi_loss = self.single_image_loss(loss_affinity, affinity_socres_label)
        return char_loss + affi_loss


class Maploss_v2(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss_v2, self).__init__()

    def batch_image_loss(self, pred_score, label_score, neg_rto, flag):

        batch_size = pred_score.shape[0]

        # positive_loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)

        positive_loss_region = pred_score * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        # negative_loss
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_score * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
                cond_flag = 0
            else:
                negative_loss = \
                    torch.sum(torch.topk(negative_loss_region.view(-1), int(neg_rto * positive_pixel_number))[0]) \
                    / (positive_pixel_number * neg_rto)
                cond_flag = 1
        else:
            # only negative pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region, 500)[0]) / 500
            cond_flag = 2

        # if flag == 'region':
        #     wandb.log({"region_positive_loss": positive_loss, "region_negative_loss": negative_loss, "region_pos_pixel_num" : positive_pixel_number, "region_neg_pixel_num" : negative_pixel_number, "region_condition":cond_flag})
        # else:
        #     wandb.log({"affi_positive_loss": positive_loss, "affi_negative_loss": negative_loss, "affi_pos_pixel_num" : positive_pixel_number, "affi_neg_pixel_num" : negative_pixel_number, "affi_condition":cond_flag})

        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(self, region_scores_label, affinity_socres_label, region_scores_pre, affinity_scores_pre, mask,
                neg_rto):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_socres_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        # loss1 = torch.sqrt(loss1 + 1e-8)
        # loss2 = torch.sqrt(loss2 + 1e-8)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.batch_image_loss(loss_region, region_scores_label, neg_rto, flag='region')
        affi_loss = self.batch_image_loss(loss_affinity, affinity_socres_label, neg_rto, flag='affinity')
        return char_loss + affi_loss


class Maploss_v2_3(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss_v2_3, self).__init__()

    def batch_image_loss(self, pred_score, label_score, neg_rto, flag, vis=False, batch_index=0):

        batch_size = pred_score.shape[0]

        # positive_loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)

        positive_loss_region = pred_score * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        # negative_loss
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_score * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
                cond_flag = 0
            else:
                negative_loss = \
                    torch.sum(torch.topk(negative_loss_region.view(-1), int(neg_rto * positive_pixel_number))[0]) \
                    / (positive_pixel_number * neg_rto)
                cond_flag = 1
        else:
            # only negative pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region, 500)[0]) / 500
            cond_flag = 2

        # if flag == 'region':
        #     wandb.log({"region_positive_loss": positive_loss, "region_negative_loss": negative_loss, "region_pos_pixel_num" : positive_pixel_number, "region_neg_pixel_num" : negative_pixel_number, "region_condition":cond_flag})
        # else:
        #     wandb.log({"affi_positive_loss": positive_loss, "affi_negative_loss": negative_loss, "affi_pos_pixel_num" : positive_pixel_number, "affi_neg_pixel_num" : negative_pixel_number, "affi_condition":cond_flag})

        total_loss = positive_loss + negative_loss

        if vis:
            positive_pixel_vis = cv2.applyColorMap((positive_pixel.reshape(-1, 384).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            negative_pixel_vis = cv2.applyColorMap((negative_pixel.reshape(-1, 384).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            positive_loss_region_vis = cv2.applyColorMap((positive_loss_region.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            negative_loss_region_vis = cv2.applyColorMap((negative_loss_region.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis_result = np.hstack([positive_pixel_vis, negative_pixel_vis, positive_loss_region_vis, negative_loss_region_vis])
            cv2.imwrite(f'/nas/home/gmuffiness/result/vis_result_inside_{flag}_pos-{positive_loss}_neg-{negative_loss}.png', vis_result)
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
            image_log = wandb.Image(vis_result, caption=f"positive loss :{positive_loss}, negative loss:{negative_loss}")
            wandb.log({f'loss_vis_inside_example_{flag}_{batch_index}': image_log})
        return total_loss

    def forward(self, region_scores_label, affinity_scores_label, region_scores_pre, affinity_scores_pre, mask,
                neg_rto, vis=False, batch_index=0):

        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_scores_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_scores_label)

        # loss1 = torch.sqrt(loss1 + 1e-8)
        # loss2 = torch.sqrt(loss2 + 1e-8)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.batch_image_loss(loss_region[4:,:,:], region_scores_label[4:,:,:], neg_rto, flag='region', vis=vis, batch_index=batch_index)
        # char_loss_synth = self.batch_image_loss(loss_region[:4,:,:], region_scores_label[:4,:,:], neg_rto, flag='region', vis=vis)
        affi_loss = self.batch_image_loss(loss_affinity[4:,:,:], affinity_scores_label[4:,:,:], neg_rto, flag='affinity', vis=vis, batch_index=batch_index)

        if vis == True:
            region_scores_pre_vis = cv2.applyColorMap((region_scores_pre.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            region_scores_label_vis = cv2.applyColorMap((region_scores_label.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            affinity_scores_pre_vis = cv2.applyColorMap((affinity_scores_pre.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            affinity_scores_label_vis = cv2.applyColorMap((affinity_scores_label.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)

            loss1_gray_vis = (loss1.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f'/nas/home/gmuffiness/result/vis_result_loss1_gray.png', loss1_gray_vis)
            loss1_vis = cv2.applyColorMap((loss1.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            loss2_vis = cv2.applyColorMap((loss2.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask_vis = cv2.applyColorMap((mask.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)

            loss_region_vis = cv2.applyColorMap((loss_region.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            loss_affinity_vis = cv2.applyColorMap((loss_affinity.reshape(-1, 384).detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)

            vis_result = np.hstack([region_scores_pre_vis, region_scores_label_vis, affinity_scores_pre_vis, affinity_scores_label_vis, loss1_vis, loss2_vis, mask_vis, loss_region_vis, loss_affinity_vis])
            cv2.imwrite(f'/nas/home/gmuffiness/result/vis_result_char-{char_loss}_affi-{affi_loss}.png', vis_result)
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
            image_log = wandb.Image(vis_result, caption=f"char loss :{char_loss}, affi_loss:{affi_loss}")
            wandb.log({f'loss_vis_example_{batch_index}':image_log})
            import ipdb; ipdb.set_trace()
        return char_loss + affi_loss

class Maploss_v3(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss_v3, self).__init__()

    def single_image_loss(self, pred_loss, label_score, neg_rto, prev_pos_pixel_number):

        # positive_loss : character region's loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pred_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        # negative_loss : background region's loss
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_loss * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
            else:
                negative_loss = \
                    torch.sum(torch.topk(negative_loss_region.view(-1), int(neg_rto * positive_pixel_number))[0]) \
                    / (neg_rto * positive_pixel_number)

        else:
            positive_pixel_number = prev_pos_pixel_number
            # only negative pixel
            positive_loss = torch.tensor(0., device='cuda:0')
            negative_loss = \
                torch.sum(torch.topk(negative_loss_region.view(-1), int(neg_rto * positive_pixel_number))[0]) \
                / (neg_rto * positive_pixel_number)
        total_loss = positive_loss + negative_loss

        return total_loss, positive_pixel_number

    def forward(self, region_scores_label, affinity_socres_label, region_scores_pre, affinity_scores_pre, mask,
                neg_rto):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_socres_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        batch_size = loss_region.shape[0]
        char_losses = torch.tensor(0, device='cuda:0')
        affi_losses = torch.tensor(0, device='cuda:0')
        pos_pixel_number = 1000

        for i in range(batch_size):
            char_loss, prev_pos_pixel_number = self.single_image_loss(loss_region[i], region_scores_label[i], neg_rto, pos_pixel_number)
            char_losses = char_losses + char_loss
            affi_loss, prev_pos_pixel_number = self.single_image_loss(loss_affinity[i], affinity_socres_label[i], neg_rto, pos_pixel_number)
            affi_losses = affi_losses + affi_loss
        # import ipdb;ipdb.set_trace()
        return (char_losses + affi_losses) / batch_size




class Maploss_v3_1(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss_v3_1, self).__init__()

    def single_image_loss(self, pred_loss, label_score, neg_rto, flag):

        # positive_loss : character region's loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pred_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        # negative_loss : background region's loss
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_loss * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
                cond_flag = 0
            else:
                negative_loss = \
                    torch.sum(torch.topk(negative_loss_region.view(-1), int(neg_rto * positive_pixel_number))[0]) \
                    / (neg_rto * positive_pixel_number)
                cond_flag = 1
        else:
            positive_loss = torch.tensor(0., device='cuda:0')
            negative_loss = torch.sum(torch.topk(negative_loss_region.view(-1), 500)[0]) / 500
            cond_flag = 2

        # if flag == 'region':
        #     wandb.log({"region_positive_loss": positive_loss, "region_negative_loss": negative_loss, "region_pos_pixel_num" : positive_pixel_number, "region_neg_pixel_num" : negative_pixel_number, "region_condition":cond_flag})
        # else:
        #     wandb.log({"affi_positive_loss": positive_loss, "affi_negative_loss": negative_loss, "affi_pos_pixel_num" : positive_pixel_number, "affi_neg_pixel_num" : negative_pixel_number, "affi_condition":cond_flag})

        total_loss = positive_loss + negative_loss

        return total_loss

    def forward(self, region_scores_label, affinity_socres_label, region_scores_pre, affinity_scores_pre, mask,
                neg_rto):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_socres_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        batch_size = loss_region.shape[0]
        char_losses = torch.tensor(0, device='cuda:0')
        affi_losses = torch.tensor(0, device='cuda:0')

        for i in range(batch_size):
            char_loss = self.single_image_loss(loss_region[i], region_scores_label[i], neg_rto, flag='region')
            char_losses = char_losses + char_loss
            affi_loss = self.single_image_loss(loss_affinity[i], affinity_socres_label[i], neg_rto, flag='affi')
            affi_losses = affi_losses + affi_loss
        # import ipdb;ipdb.set_trace()
        return (char_losses + affi_losses) / batch_size
