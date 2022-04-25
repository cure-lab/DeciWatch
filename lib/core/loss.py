import torch
import torch.nn as nn
import numpy as np
import math
import cv2
import torch.nn.functional as F
from lib.models.smpl import SMPL
from lib.utils.geometry_utils import *


class DeciWatchLoss(nn.Module):

    def __init__(self, w_denoise, lamada, smpl_model_dir, smpl):
        super().__init__()
        self.w_denoise = w_denoise
        self.lamada = lamada
        self.smpl_model_dir = smpl_model_dir
        self.smpl = smpl

    def mask_lr1_loss(self, inputs, mask, targets):
        Bs, C, L = inputs.shape

        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)
        loss = F.l1_loss(
            inputs * not_mask, targets * not_mask, reduction='sum') / N
        return loss

    def forward(self,
                recover,
                denoise,
                gt,
                mask_src,
                mask_pad,
                use_smpl_loss=False):
        if use_smpl_loss == True and self.smpl == True:
            return self.forward_smpl(recover, denoise, gt, mask_src, mask_pad)
        else:
            return self.forward_lr1(recover, denoise, gt, mask_src, mask_pad)

    def forward_lr1(self, recover, denoise, gt, mask_src, mask_pad):
        B, L, C = recover.shape
        recover = recover.permute(0, 2, 1)
        denoise = denoise.permute(0, 2, 1)  #[b,c,t]
        gt = gt.permute(0, 2, 1)

        loss_denoise = self.mask_lr1_loss(denoise, mask_src, gt)  #mask:[b, t]
        loss_pose = self.mask_lr1_loss(recover, mask_pad, gt)

        weighted_loss = self.w_denoise * loss_denoise + self.lamada * loss_pose

        return weighted_loss

    def forward_smpl(self, recover, denoise, gt, mask_src, mask_pad):
        SMPL_TO_J14 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 38]
        B, L, C = recover.shape

        recover = rot6D_to_axis(recover.reshape(-1, 6)).reshape(-1, 24 * 3)
        denoise = rot6D_to_axis(denoise.reshape(-1, 6)).reshape(-1, 24 * 3)
        gt = rot6D_to_axis(gt.reshape(-1, 6)).reshape(-1, 24 * 3)

        device = recover.device
        smpl = SMPL(model_path=self.smpl_model_dir,
                    gender="neutral",
                    batch_size=1).to(device)

        gt_smpl_joints = smpl.forward(
            global_orient=gt[:, 0:3].to(torch.float32),
            body_pose=gt[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        denoise_smpl_joints = smpl.forward(
            global_orient=denoise[:, 0:3].to(torch.float32),
            body_pose=denoise[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        recover_smpl_joints = smpl.forward(
            global_orient=recover[:, 0:3].to(torch.float32),
            body_pose=recover[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        gt_smpl_joints = gt_smpl_joints.reshape(B, L, -1).permute(0, 2, 1)
        denoise_smpl_joints = denoise_smpl_joints.reshape(B, L,
                                                          -1).permute(0, 2, 1)
        recover_smpl_joints = recover_smpl_joints.reshape(B, L,
                                                          -1).permute(0, 2, 1)

        loss_denoise = self.mask_lr1_loss(denoise_smpl_joints, mask_src,
                                          gt_smpl_joints)  #mask:[b, t]
        loss_pose = self.mask_lr1_loss(recover_smpl_joints, mask_pad,
                                       gt_smpl_joints)

        weighted_loss = self.w_denoise * loss_denoise + self.lamada * loss_pose

        return weighted_loss
