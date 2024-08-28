import torch
import torch.nn as nn
import torch.nn.functional as F

# from .swin_transformer import SwinTransformer

import numpy as np
from .network_unet import UNetRes
from contrast.MIRnet import MIRNet
from contrast.Restormer import Restormer
# from .Mambalayer import MambaLayer
from contrast.sun_unet0522 import UNet
from contrast.MPRNet import MPRNet
from contrast.SRMNet import SRMNet
from contrast.SPDNet import Rainnet
from contrast.FSNet import FSNet
from PIL import Image



class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super(NewCRFDepth,self).__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        # self.half1 = B.conv(3,24,mode='C'+'G')
        # self.down1 = B.sequential(*[B.conv(24, 24, mode='C'+'G') for _ in range(2)], B.downsample_strideconv(24, 48, mode='2'+'G'))
        

        # backbone_cfg = dict(
        #     embed_dim=embed_dim,
        #     depths=depths,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     ape=False,
        #     drop_path_rate=0.3,
        #     patch_norm=True,
        #     use_checkpoint=False,
        #     frozen_stages=frozen_stages
        #     self.unet = UNetRes()
        # self.Restormer = Uformer(img_size=256, embed_dim=16,depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        #          win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
        self.Restormer = FSNet()
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        

    
    def forward(self,rgbw_data):
        # 输进来的imgs大小为（batchsize,channels,w,h）
        img_MASK=self.RGB_TO_RGBMASK(rgbw_data)
        w=self.RGB_to_W(rgbw_data)
        # 合并
        rgbw=torch.cat((img_MASK,w),1)
        feats = self.Restormer(rgbw)
        return feats[:,0:3,]
        # return feats,w

    def RGB_to_W(self,imgs):
        H,W=imgs.shape[2],imgs.shape[3]
        R = imgs[:, 0, :, :]  # 第一个通道对应于 R
        G = imgs[:, 1, :, :]  # 第二个通道对应于 G
        B = imgs[:, 2, :, :]  # 第三个通道对应于 B
        w=(R/3.0+G/3.0+B/3.0)
        #这里w的形状为Size([batchsize, h, w])，所以插入一个通道维度
        # print(w.shape)
        w = w.unsqueeze(1)
        #对w加入mask,使它有缺失值
        mosaicked=np.zeros((H, W))  
        mosaicked[::2,::2]=1
        mosaicked[1::2,1::2]=1
        
        mosaicked_tensor = torch.from_numpy(mosaicked).float().cuda().unsqueeze(0).unsqueeze(0)
        w = w*mosaicked_tensor 
        w = self.gaussian(w,mosaicked_tensor,H,W,num_channels=1)
#######################################################双线性插值模块      
        # HG_kernel=torch.tensor([[0,1,0],[1,4,1],[0,1,0]])/4
        # HG_kernel=HG_kernel.unsqueeze(0).unsqueeze(0)
        # # 假设 HG_kernel 是一个 PyTorch 张量，将其数据类型转换为与 w 相同
        # HG_kernel = HG_kernel.to(w.device)
        # #送入卷积，使之称为没有缺失值的W,经过映射的input数值输入
        # dst = torch.conv2d(input=w,weight=HG_kernel,padding=1,stride=1,bias=None)
        # # dst = torch.cat([dst] * 3, dim=1)
###########################################################添加噪声
        # dst = add_gaussian_noise(dst)
###################################################### MIRnet处理缺失值问题
        # w = torch.cat([w]*3,dim=1)
        # dst = self.model_restoration(w)
        # dst = torch.clamp(dst,0,1) 
###############################################保存到本地看一下
        # dst = dst.cpu().squeeze().numpy()
        # dst=(dst*255).astype('uint8')
        # print(dst.shape)
        # dst = Image.fromarray(dst, mode='L')
        # dst.save('w_fill.png')
        # w = torch.cat([w] * 3,dim=1)
        return w
    def RGB_TO_RGBMASK(self,imgs):
        # 创建三个矩阵 cfa_mask0, cfa_mask1, cfa_mask2，每个都是大小为 H x W
        H,W=imgs.shape[2],imgs.shape[3]
        cfa_mask0 = np.zeros((H, W))
        cfa_mask1 = np.zeros((H, W))
        cfa_mask2 = np.zeros((H, W))
        ####################################################################################
        cfa_mask0[0::2,0::2]=1
        cfa_mask0[2::4,3::4]=1
        cfa_mask0[3::4,2::4]=1
        cfa_mask1[0::4,3::4]=1
        cfa_mask1[1::4,2::4]=1
        cfa_mask1[2::4,1::4]=1
        cfa_mask1[3::4,0::4]=1
        cfa_mask2[0::4,1::4]=1
        cfa_mask2[1::4,0::4]=1


        # 合并三个矩阵成一个 [3, H, W] 的矩阵
        cfa_mask_combined = np.stack((cfa_mask0, cfa_mask1, cfa_mask2), axis=0)
        cfa_mask_combined=torch.from_numpy(cfa_mask_combined).cuda().unsqueeze(0)
        # 可以执行逐元素相乘
        imgs = (imgs * cfa_mask_combined).float()/3.0
        rgb = self.gaussian(imgs,cfa_mask_combined,H,W,num_channels=3)
##################保存模块 
        # # 将张量转换为NumPy数组
        # rgb_np=rgb[0,:,:,:]
        # rgb_np = rgb_np.squeeze(0).cpu().numpy()  # 假设只有一个样本，去除批次维度
        # # 转换数值范围为 [0, 1] 到 [0, 255]
        # rgb_np = (rgb_np * 255).astype(np.uint8)
        # # 创建一个PIL图像对象
        # image = Image.fromarray(rgb_np.transpose(1, 2, 0))  # 调整通道顺序
        # # 保存图像到本地文件
        # image.save("output3.jpg")
        return rgb
    def gaussian(self,imgs,cfa_mask_combined,H,W,num_channels):
#####################################高斯处理模块
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

##################################tiny 24 ,48, 96, 192, 384, 768,
#################################small 24 ,48, 96, 192, 384, 768
##################################base 32, 64,128, 256, 512, 1024
# class UNet(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[24 ,48, 96, 192, 384, 768], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
#         super(UNet, self).__init__()

#         self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

#         self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
#         self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
#         self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))
#         self.m_down4 = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[3], nc[4], mode='2'+act_mode))
#         self.m_down5 = B.sequential(*[B.conv(nc[4], nc[4], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[4], nc[5], mode='2'+act_mode))

#         self.m_body  = B.sequential(*[B.conv(nc[5], nc[5], mode='C'+act_mode) for _ in range(nb+1)])

#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

#         self.m_up3 = B.sequential(upsample_block(nc[5], nc[4], mode='2'+act_mode), *[B.conv(nc[4], nc[4], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[4], nc[3], mode='2'+act_mode), *[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up0 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
#         self.m_up00 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])
#         self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

#     def forward(self, x0):
#         outs=[]
#         x1 = self.m_head(x0)
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         x5 = self.m_down4(x4)
#         x6 = self.m_down5(x5)
#         x = self.m_body(x6)
#         outs.append(x)
#         x = self.m_up3(x+x6)
#         outs.append(x)
#         x = self.m_up2(x+x5)
#         outs.append(x)
#         x = self.m_up1(x+x4)
#         outs.append(x)
#         x = self.m_up0(x+x3)
#         x = self.m_up00(x+x2)
#         x = self.m_tail(x+x1) + x0
#         outs = outs[::-1]
#         return tuple(outs)


    

def add_gaussian_noise(imgs, mean=0, std=0.8):
    """
    Add Gaussian noise to the input images.
    
    Args:
        imgs (torch.Tensor): Input images tensor with shape (batch_size, channels, height, width).
        mean (float): Mean of the Gaussian noise distribution.
        std (float): Standard deviation of the Gaussian noise distribution.
    
    Returns:
        torch.Tensor: Noisy images tensor with the same shape as input.
    """
    noise = torch.randn_like(imgs) * std + mean
    noisy_imgs = imgs + noise
    return noisy_imgs
