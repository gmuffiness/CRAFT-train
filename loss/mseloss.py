import torch
import torch.nn as nn
import numpy as np


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
            negative_loss = torch.sum(
                torch.topk(
                    negative_loss_region.view(-1), int(3 * positive_pixel_number)
                )[0]
            ) / (positive_pixel_number * 3)

        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(
        self,
        region_scores_label,
        affinity_socres_label,
        region_scores_pre,
        affinity_scores_pre,
        mask,
    ):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert (
            region_scores_label.size() == region_scores_pre.size()
            and affinity_socres_label.size() == affinity_scores_pre.size()
        )
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

    def single_image_loss(self, pre_loss, loss_label, neg_rto):

        batch_size = pre_loss.shape[0]

        # positive_loss
        positive_pixel = (loss_label > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pre_loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        # negative_loss
        negative_pixel = (loss_label <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pre_loss * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
            else:
                negative_loss = torch.sum(
                    torch.topk(
                        negative_loss_region.view(-1),
                        int(neg_rto * positive_pixel_number),
                    )[0]
                ) / (positive_pixel_number * neg_rto)

        else:
            # only negative pixel
            negative_loss = torch.sum(torch.topk(negative_loss_region, 500)[0]) / 500

        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(
        self,
        region_scores_label,
        affinity_socres_label,
        region_scores_pre,
        affinity_scores_pre,
        mask,
        neg_rto,
    ):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert (
            region_scores_label.size() == region_scores_pre.size()
            and affinity_socres_label.size() == affinity_scores_pre.size()
        )
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        # loss1 = torch.sqrt(loss1 + 1e-8)
        # loss2 = torch.sqrt(loss2 + 1e-8)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_region, region_scores_label, neg_rto)
        affi_loss = self.single_image_loss(
            loss_affinity, affinity_socres_label, neg_rto
        )
        return char_loss + affi_loss

class Maploss_v3(nn.Module):
    def __init__(self, use_gpu=True):

        super(Maploss_v3, self).__init__()

    def single_image_loss(self, pre_loss, loss_label, neg_rto, n_min_neg):

        batch_size = pre_loss.shape[0]

        positive_loss, negative_loss = 0, 0
        for single_loss, single_label in zip(pre_loss, loss_label):

            # positive_loss
            pos_pixel = (single_label >= 0.1).float()
            n_pos_pixel = torch.sum(pos_pixel)
            pos_loss_region = single_loss * pos_pixel
            positive_loss += torch.sum(pos_loss_region) / max(n_pos_pixel, 1e-12)

            # negative_loss
            neg_pixel = (single_label < 0.1).float()
            n_neg_pixel = torch.sum(neg_pixel)
            neg_loss_region = single_loss * neg_pixel

            if n_pos_pixel != 0:
                if n_neg_pixel < neg_rto * n_pos_pixel:
                    negative_loss += torch.sum(neg_loss_region) / n_neg_pixel
                else:
                    n_hard_neg = max(n_min_neg, neg_rto*n_pos_pixel)
                    #n_hard_neg = neg_rto*n_pos_pixel
                    negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), int(n_hard_neg))[0]) / n_hard_neg
            else:
                #only negative pixel
                negative_loss += torch.sum(torch.topk(neg_loss_region.view(-1), n_min_neg)[0]) / n_min_neg

        total_loss = (positive_loss + negative_loss) / batch_size

        return total_loss

    def forward(self, region_scores_label, affinity_scores_label, region_scores_pre, affinity_scores_pre, mask, neg_rto, n_min_neg):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_scores_label.size() == region_scores_pre.size() and affinity_scores_label.size() == affinity_scores_pre.size()
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_scores_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_region, region_scores_label, neg_rto, n_min_neg)
        affi_loss = self.single_image_loss(loss_affinity, affinity_scores_label, neg_rto, n_min_neg)

        return char_loss + affi_loss


#
# class Maploss(nn.Module):
#     def __init__(self, use_gpu = True):
#
#         super(Maploss,self).__init__()
#
#     def single_image_loss(self, pre_loss, loss_label):
#         batch_size = pre_loss.shape[0]
#         sum_loss = torch.mean(pre_loss.view(-1))*0
#         pre_loss = pre_loss.view(batch_size, -1)
#         loss_label = loss_label.view(batch_size, -1)
#         internel = batch_size
#         for i in range(batch_size):
#             average_number = 0
#             loss = torch.mean(pre_loss.view(-1)) * 0
#             positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
#             average_number += positive_pixel
#             if positive_pixel != 0:
#                 posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
#                 sum_loss += posi_loss
#                 if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
#                     nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
#                     average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
#                 else:
#                     nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
#                     average_number += 3*positive_pixel
#                 sum_loss += nega_loss
#             else:
#                 nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
#                 average_number += 500
#                 sum_loss += nega_loss
#             #sum_loss += loss/average_number
#
#
#         # import math
#         # if sum_loss > 1e8 or math.isnan(sum_loss):
#         #     import ipdb;ipdb.set_trace()
#
#
#         return sum_loss
#
#
#
#     def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
#
#         gh_label = gh_label
#         gah_label = gah_label
#         p_gh = p_gh
#         p_gah = p_gah
#         loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
#
#         assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
#         loss1 = loss_fn(p_gh, gh_label)
#         loss2 = loss_fn(p_gah, gah_label)
#         loss_g = torch.mul(loss1, mask)
#         loss_a = torch.mul(loss2, mask)
#
#         char_loss = self.single_image_loss(loss_g, gh_label)
#         affi_loss = self.single_image_loss(loss_a, gah_label)
#
#
#         return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]
