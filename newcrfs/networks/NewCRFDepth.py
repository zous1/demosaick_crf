import torch
import torch.nn as nn
import torch.nn.functional as F

# from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .rlfn_ntire import RLFN_Prune
from .network_unet import UNetRes
import networks.basicblock as B
from .SAN import NONLocalBlock2D
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

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        elif version[:-2] =='small':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        elif version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]

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
        # )

        embed_dim = 128
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )
######################################修改了backbone部分,改成了unet+mamba
        # self.backbone = SwinTransformer(**backbone_cfg)
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf1 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[1], window_size=win, v_dim=768, num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[0], window_size=win, v_dim=64,  num_heads=4)
        self.crf00= NewCRF(input_dim=in_channels[1], embed_dim=64,          window_size=win, v_dim=32,  num_heads=2)
        self.crf000=NewCRF(input_dim=in_channels[0], embed_dim=32,          window_size=win, v_dim=16,  num_heads=2)
        #数值
        #self.crf3=NewCRF(input_dim=1536,embed_dim=1024,windows_size=7,v_dim=512,num_heads=32)
        #self.crf2=NewCRF(input_dim=768,embed_dim=512,windows_size=7,v_dim=256,num_heads=16)
        #self.crf2=NewCRF(input_dim=384,embed_dim=256,windows_size=7,v_dim=128,num_heads=8)
        #self.crf2=NewCRF(input_dim=192,embed_dim=128,windows_size=7,v_dim=64,num_heads=4)
        
        # self.disp_head1 = DispHead(input_dim=crf_dims[0])
        self.disp_head1 = DispHead(input_dim=32)
##################################################
        # self.decoder = PSP(**decoder_cfg)
        # self.unet = UNet()
        # self.ECA = eca_layer(channel = 4)
        self.aggregation = NONLocalBlock2D(in_channels=in_channels[3],inter_channels=384, sub_sample=False,bn_layer=True)
        # self.aggregation = PAFPN(FPN)
#######################################模块mamba
        self.unet_mamba=UNet_mamba()
        self.up_mode = 'bilinear'
        # if self.up_mode == 'mask':
        #     self.mask_head = nn.Sequential(
        #         nn.Conv2d(crf_dims[0], 64, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(64, 16*9, 1, padding=0))


        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        
#############################添加一个MIR的初始化
        # self.model_restoration = UNetRes()
        # self.model_restoration.init_weights(pretrained=pretrained[1])
        
        # self.backbone.init_weights(pretrained=pretrained[0])
        # self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    
    def forward(self, imgs):
        #输进来的imgs大小为（batchsize,channels,w,h）
        img_MASK=self.RGB_TO_RGBMASK(imgs)
        w=self.RGB_to_W(imgs)
        #合并
        rgbw=torch.cat((img_MASK,w),1)
        ####残差通道模块
        # rgbw = self.ECA(rgbw)
        #用mamba作为编码器，对rgbw进行特征提取
        feats = self.unet_mamba(rgbw)  
        # 输出元组里每个张量的大小
        # for i, feat in enumerate(feats):
        #     print(f"Tensor {i} shape: {feat.shape}")
        #将经过encoder处理的feats数据送入psp模块
        # ppm_out = self.decoder(feats)
        # ppm_out = self.non_local(feats[3])
        ppm_out = self.aggregation(feats[3])
        e3 = self.crf1(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        
       
        e2 = self.crf0(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2) 
         
       
        e1 = self.crf00(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        
      
        e0 = self.crf000(feats[0], e1)
        
        d1 = self.disp_head1(e0,4)
        depth = d1 + img_MASK
        return depth,w
    #增加了rgb转W的函数
    def RGB_to_W(self,imgs):
        H,W=imgs.shape[2],imgs.shape[3]
        R = imgs[:, 0, :, :]  # 第一个通道对应于 R
        G = imgs[:, 1, :, :]  # 第二个通道对应于 G
        B = imgs[:, 2, :, :]  # 第三个通道对应于 B
        w=(R/3.0+G/3.0+B/3.0)
        #这里w的形状为Size([8, 384, 384])，所以加一维度
        # print(w.shape)
        w = w.unsqueeze(1)
        #对w加入mask,使它有缺失值
        mosaicked=np.zeros((H, W))  
        mosaicked[::2,::2]=1
        mosaicked[1::2,1::2]=1
        
        mosaicked_tensor = torch.from_numpy(mosaicked).float().cuda().unsqueeze(0).unsqueeze(0)
        w=w*mosaicked_tensor   
        ##################################检查有缺失值的w
        # w_cpu=w.cpu().squeeze().numpy()
        # mapped_w_cpu=(w_cpu*255).astype('uint8')
        # mapped_w_cpu = Image.fromarray(mapped_w_cpu, mode='L')
        # mapped_w_cpu.save('w.png')
#######################################################双线性插值模块
        
        HG_kernel=torch.tensor([[0,1,0],[1,4,1],[0,1,0]])/4
        HG_kernel=HG_kernel.unsqueeze(0).unsqueeze(0)
        # 假设 HG_kernel 是一个 PyTorch 张量，将其数据类型转换为与 w 相同
        HG_kernel = HG_kernel.to(w.device)
        #送入卷积，使之称为没有缺失值的W,经过映射的input数值输入
        dst = torch.conv2d(input=w,weight=HG_kernel,padding=1,stride=1,bias=None)
        # dst = torch.cat([dst] * 3, dim=1)
###########################################################添加噪声
        # dst = add_gaussian_noise(dst)
###############################################保存到本地看一下
        # dst = dst.cpu().squeeze().numpy()
        # dst=(dst*255).astype('uint8')
        # print(dst.shape)
        # dst = Image.fromarray(dst, mode='L')
        # dst.save('w_fill.png')
        # w = torch.cat([w] * 3,dim=1)
        return dst
    def RGB_TO_RGBMASK(self,imgs):
        # 创建三个矩阵 cfa_mask0, cfa_mask1, cfa_mask2，每个都是大小为 H x W
        H,W=imgs.shape[2],imgs.shape[3]
        cfa_mask0 = np.zeros((H, W))
        cfa_mask1 = np.zeros((H, W))
        cfa_mask2 = np.zeros((H, W))
        ####################################################################################
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
#####################################高斯处理模块，1101添加
        # 初始化带有RGB通道的高斯权重
        gaussian_weight_size=4
        num_channels=3
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
##################保存模块 
        # # 将张量转换为NumPy数组
        # rgb_np=rgb[0,:,:,:]
        # rgb_np = rgb_np.squeeze(0).cpu().numpy()  # 假设只有一个样本，去除批次维度
        # # 转换数值范围为 [0, 1] 到 [0, 255]
        # rgb_np = (rgb_np * 255).astype(np.uint8)
        # # 创建一个PIL图像对象
        # print(rgb_np.shape)img
        # image = Image.fromarray(rgb_np.transpose(1, 2, 0))  # 调整通道顺序
        # # 保存图像到本地文件
        # image.save("output3.jpg")
        return rgb
class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        #修改了输出通道数
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
#####################################################超分
        self.superRE = RLFN_Prune(in_channels=3,out_channels=3)

    def forward(self, x1, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x1))
        if scale > 1:
#####################################这里scale传进来的是4，但是没用到scale
            x = self.superRE(x)
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
###################################加了一个残差的连接
        return x + (x * y.expand_as(x))

class UNet_mamba(nn.Module):
    def __init__(self, in_nc=4, nc=[96,192,384,768],nb=2, act_mode='R', downsample_mode='strideconv',):
        super(UNet_mamba, self).__init__()
        #更改了m_head的处理。更成了先加conv+mamba
        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])
        # self.m_head = B.sequential(B.conv(in_nc, nc[0], mode='C'+act_mode[-1]),MambaLayer(nc[0]))
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))
        # self.m_down4 = B.sequential(MambaLayer(nc[3]), downsample_block(nc[3], nc[4], mode='2'+act_mode))
        # self.m_down5 = B.sequential(MambaLayer(nc[4]), downsample_block(nc[4], nc[5], mode='2'+act_mode))


    def forward(self, x0):
        outs=[]
        x1 = self.m_head(x0)
        outs.append(x1)
        x2 = self.m_down1(x1)
        outs.append(x2)
        x3 = self.m_down2(x2)
        outs.append(x3)
        x4 = self.m_down3(x3)
        outs.append(x4)
        return tuple(outs)

    
#####################################这里scale默认是2，但是传进来的是4
def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


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
