from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

# 读取图像文件
image_path =  r"/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/demosaick_crf/datasets/nyu/output/pred_0.png"  # 替换为你的图像路径
imgs = Image.open(image_path).convert('RGB')  # 转换为RGB格式
imgs = transforms.ToTensor()(imgs).cuda()  # 转换为张量并移动到GPU

# 获取图像的高度和宽度
H, W = imgs.shape[1], imgs.shape[2]

# 创建三个矩阵 cfa_mask0, cfa_mask1, cfa_mask2
cfa_mask0 = np.zeros((H, W))
cfa_mask1 = np.zeros((H, W))
cfa_mask2 = np.zeros((H, W))

# 根据CFA模式填充掩码矩阵
cfa_mask0[2::4, 3::4] = 1
cfa_mask0[3::4, 2::4] = 1
cfa_mask1[0::4, 3::4] = 1
cfa_mask1[1::4, 2::4] = 1
cfa_mask1[2::4, 1::4] = 1
cfa_mask1[3::4, 0::4] = 1
cfa_mask2[0::4, 1::4] = 1
cfa_mask2[1::4, 0::4] = 1

# 合并三个矩阵成一个 [3, H, W] 的矩阵
cfa_mask_combined = np.stack((cfa_mask0, cfa_mask1, cfa_mask2), axis=0)
cfa_mask_combined = torch.from_numpy(cfa_mask_combined).cuda().unsqueeze(0)

# 对输入的图像应用CFA mask
imgs_masked = imgs * cfa_mask_combined.float().squeeze(0)
torch.save(cfa_mask_combined,"mask.png")
# 保存图像模块
rgb_np = imgs_masked.cpu().numpy()  # 将张量转换为NumPy数组
# 转换数值范围为 [0, 1] 到 [0, 255]
rgb_np = (rgb_np * 255).astype(np.uint8)
# 创建一个PIL图像对象
image = Image.fromarray(rgb_np.transpose(1, 2, 0))  # 调整通道顺序
# 保存图像到本地文件
image.save("output4.png")
