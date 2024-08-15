from __future__ import print_function
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
from itertools import cycle
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init
from torch.nn.functional import interpolate, softmax
from torch.nn import Parameter
import numpy as np
import math
# from GaussianModel_0424 import GaussianModel as GaussianModel_for_inpaintRGB
from torchvision.transforms import ToTensor
from PIL import Image

import cv2

# from common import flip

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class LISTAConvDict(nn.Module):

    def __init__(self):

        super(LISTAConvDict, self).__init__()

        
        #-------------------
        # RGB重建网络的
        self.layer_RGB = UNet()


    def step2_detach_rggb(self, rgbw_data, rgb_mask):
        
        mask_comp_step2 = rgb_mask
        H,W=rgbw_data.shape[2], rgbw_data.shape[3]
        # GaussianFilter__ = GaussianModel_for_inpaintRGB(gaussian_weight_size=4, num_channels=3)
        # RGB = GaussianFilter__(RGB, mask_comp_step2)
        rgbw_data = self.gaussian(rgbw_data, mask_comp_step2,H,W,num_channels=3)
####################################################################
        # 将张量转换为NumPy数组
        rgb_np=rgbw_data[0,:,:,:]
        rgb_np = rgb_np.squeeze(0).cpu().numpy()  # 假设只有一个样本，去除批次维度
        # 转换数值范围为 [0, 1] 到 [0, 255]
        rgb_np = (rgb_np * 255).astype(np.uint8)
        # 创建一个PIL图像对象
        # print(rgb_np.shape)
        image = Image.fromarray(rgb_np.transpose(1, 2, 0))  # 调整通道顺序
        # 保存图像到本地文件
        image.save("output3.jpg")
        return rgbw_data[:, :1, :, :].float(), rgbw_data[:, 1:2, :, :].float(), rgbw_data[:, 2:, :, :].float()



    def RGB_TO_RGBMASK(self, rgb):
        
        # gt_data = gt_data.astype(np.float64) / 255.
        # rgb = rgb / 3.
       
        # 
        batch, channel, height, width = rgb.shape

        # （3）
        # kodak_cfa = torch.zeros(batch, channel, height, width)
        kodak_cfa = np.zeros((batch, channel, height, width))
        # kodak_cfa = np.zeros((height, width, 3))
        # 赋值R
        kodak_cfa[:, 0, 3::4, 2::4] = 1
        kodak_cfa[:, 0, 2::4, 3::4] = 1
        
        # 赋值G   
        kodak_cfa[:, 1, 0::4, 3::4] = 1
        kodak_cfa[:, 1, 1::4, 2::4] = 1
        kodak_cfa[:, 1, 2::4, 1::4] = 1
        kodak_cfa[:, 1, 3::4, 0::4] = 1
        
        # 赋值B
        kodak_cfa[:, 2, 1::4, 0::4] = 1
        kodak_cfa[:, 2, 0::4, 1::4] = 1
        
        # kodak_cfa = kodak_cfa.cuda()
        cfa_mask_combined=torch.from_numpy(kodak_cfa).cuda()
        print(kodak_cfa[0,0,2,0:9])
        rgb_inRGBW = (rgb * cfa_mask_combined).float()/3.0
        
        R, G, B = self.step2_detach_rggb(rgb_inRGBW, cfa_mask_combined)
        # R = R.cuda()
        # G = G.cuda()
        # B = B.cuda()
        
        return R, G, B
    

    def gaussian(self,imgs,cfa_mask_combined,H,W,num_channels):
        # 初始化带有RGB通道的高斯权重
        gaussian_weight_size=4
        weight = torch.normal(mean=1, std=(gaussian_weight_size/8), 
                            size=(gaussian_weight_size, gaussian_weight_size), device=torch.device("cuda"))
        weight = weight[None, None, :, :]
        Gaussian_weight = torch.cat([weight] * num_channels, dim=0)
        #raw直接为经过cfa处理的有缺失值的rgb
        rgb_raw=imgs
        # 在每个RGB通道上进行卷积操作
##################################这里修改padding，padding=gaussian_weight_size // 2-->padding=1
        rgb_Gaussian = F.conv2d(input=rgb_raw, weight=Gaussian_weight, stride=1,
                                padding=gaussian_weight_size // 2, groups=num_channels, bias=None)
        rgb_Gaussian = rgb_Gaussian[:, :, :H, :W]
        # 对掩码也进行卷积
        mask_4 = cfa_mask_combined.to(torch.float32)
        mask_Gaussian = F.conv2d(input=mask_4, weight=Gaussian_weight, stride=1,
                                padding=gaussian_weight_size // 2, groups=num_channels, bias=None)
        mask_Gaussian=mask_Gaussian[:,:,:H,:W]
        epsilon = 0.1 / 255
        rgb = rgb_Gaussian / (mask_Gaussian + epsilon)
        
        # 将原始像素值赋值回去
        index = torch.where(rgb_raw != 0)
        data = rgb_raw
        temp = data[index]
        rgb[index] = temp   
        return rgb  
        
    def bic_interpolation_W(self, w_inputs):
        
        HG_kernel = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32) / 4
        HG_kernel = HG_kernel.unsqueeze(0).unsqueeze(0)
        HG_kernel = HG_kernel.cuda()

        out = F.conv2d(input=w_inputs, weight=HG_kernel, stride=1, padding=1, bias=None)
        
        return out.float()
    
    
    def rgb_to_w(self, imgs):
        R = imgs[:, 0, :, :]  # 第一个通道对应于 R
        G = imgs[:, 1, :, :]  # 第二个通道对应于 G
        B = imgs[:, 2, :, :]  # 第三个通道对应于 B
        w_gt = (R + G + B) / 3.0
        w_gt = w_gt.unsqueeze(1)
        
        b, c, h, w = imgs.shape
        w_mask = torch.zeros(b, 1, h, w)
        w_mask[:, :, ::2, ::2] = 1.
        w_mask[:, :, 1::2, 1::2] = 1.
        w_mask = w_mask.cuda()
        w_gt = w_gt * w_mask
        
        
        # w_gt = torch.cat([w_gt] * 3, dim=1)
        return w_gt



    def forward(self, rgbw_data):
        
        R, G, B = self.RGB_TO_RGBMASK(rgbw_data)
        
        # print('--------------217', rgbw_data.shape)
        w = self.rgb_to_w(rgbw_data)

        w_data = self.bic_interpolation_W(w)

        
        # w_data = outputs.clone()

        # R, G, B = self.step2_detach_rggb(rgbw_data, rgb_mask)
        # R = R.cuda()
        # G = G.cuda()
        # B = B.cuda()
        #---------------------
        # RGB重建网络的

        input_tensor = torch.cat((R, G, B, w_data), dim=1)
        output_RGB = self.layer_RGB(input_tensor)

        return output_RGB
    