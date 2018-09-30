# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamRPNBIG(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )

        self.conv_reg1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_reg2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.reg1_kernel = []
        self.cls1_kernel = []

    def forward(self, x):
        x_f = self.featureExtract(x)

        batch_size = x_f.size(0)
        reg_conv_output = self.conv_reg2(x_f)
        cls_conv_output = self.conv_cls2(x_f)

        cls_corr_list = []
        reg_corr_list = []
        for i_batch in range(batch_size):
            i_cls_corr = F.conv2d(torch.unsqueeze(cls_conv_output[i_batch], 0), self.cls1_kernel[i_batch])
            cls_corr_list.append(i_cls_corr)
            i_reg_corr = F.conv2d(torch.unsqueeze(reg_conv_output[i_batch], 0), self.reg1_kernel[i_batch])
            i_reg_corr = self.regress_adjust(i_reg_corr)
            reg_corr_list.append(i_reg_corr)

        cls_corr = torch.stack(cls_corr_list, dim=0)
        cls_corr = torch.squeeze(cls_corr)
        reg_corr = torch.stack(reg_corr_list, dim=0)
        reg_corr = torch.squeeze(reg_corr)

        """
        # return tensors of shape (17,17,4K) and (17,17,2k), respectively 
        return self.regress_adjust(
			F.conv2d(self.conv_reg2(x_f), self.reg1_kernel)
			), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)
        """
        return reg_corr, cls_corr # of shape (50, 4K, 17, 17), (50, 2K, 17, 17)
	

    def temple(self, z):
        z_f = self.featureExtract(z)
        reg1_kernel_raw = self.conv_reg1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = reg1_kernel_raw.data.size()[-1]

        self.reg1_kernel = reg1_kernel_raw.view(-1, self.anchor*4, self.feature_out, kernel_size, kernel_size)#50, 4K, 512, 4, 4
        self.cls1_kernel = cls1_kernel_raw.view(-1, self.anchor*2, self.feature_out, kernel_size, kernel_size)#50, 2K, 512, 4, 4
