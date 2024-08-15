from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
# 读取图像文件
image_path = r"F:/LuYD/Swin-Transformer-main/NeWCRFs-master/output_image.png"  # 替换为你的图像路径
input_image = Image.open(image_path)

# 定义转换操作
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])

# 对图像进行转换
img_tensor = preprocess(input_image)

# 显示张量的形状和数据类型
print("Tensor Shape:", img_tensor.shape)
print("Tensor Data Type:", img_tensor.dtype)

# 存储在imgs中
imgs = img_tensor.unsqueeze(0)  # 添加一个维度，成为1张图片的张量数据
# 创建三个矩阵 cfa_mask0, cfa_mask1, cfa_mask2，每个都是大小为 2040x1360
H,W=imgs.shape[3],imgs.shape[2]
cfa_mask0 =cfa_mask1=cfa_mask2= np.zeros((H, W))
####################################################################################1114没改完
cfa_mask0[2::4,3::4]=1
cfa_mask0[3::4,2::4]=1
cfa_mask1[0::4,3::4]=1
cfa_mask1[1::4,2::4]=1
cfa_mask1[2::4,1::4]=1
cfa_mask1[3::4,0::4]=1
cfa_mask2[0::4,1::4]=1
cfa_mask2[1::4,0::4]=1

# 合并三个矩阵成一个 [3, 2040, 1360] 的矩阵
cfa_mask_combined = np.stack((cfa_mask0, cfa_mask1, cfa_mask2), axis=0)
cfa_mask_combined=torch.from_numpy(cfa_mask_combined).cuda().unsqueeze(0)
# 可以执行逐元素相乘
print(imgs.shape)
imgs = imgs.permute(0, 1, 3, 2)
print(cfa_mask_combined.shape)
# 将imgs移到GPU上
imgs = imgs.cuda()

# 执行逐元素相乘（确保两个张量形状相同）
imgs = (imgs * cfa_mask_combined.float()).float()
print(imgs[0,0,:,:])
print(imgs[0,1,:,:])
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
rgb_Gaussian = rgb_Gaussian[:, :, :2040, :1360]
# 对掩码也进行卷积
mask_4 = cfa_mask_combined.to(torch.float32)
mask_Gaussian = F.conv2d(input=mask_4, weight=Gaussian_weight, stride=1,
                        padding=gaussian_weight_size // 2, groups=num_channels, bias=None)
mask_Gaussian=mask_Gaussian[:,:,:2040,:1360]
epsilon = 0.1 / 255
rgb = rgb_Gaussian / (mask_Gaussian + epsilon)

# 将原始像素值赋值回去
index = torch.where(rgb_raw != 0)
data = rgb_raw
temp = data[index]
rgb[index] = temp

rgb = rgb.permute(0, 1, 3, 2)   
##################保存模块 
# 将张量转换为NumPy数组
rgb_np=rgb[0,:,:,:]
rgb_np = rgb_np.squeeze(0).cpu().numpy()  # 假设只有一个样本，去除批次维度
# 转换数值范围为 [0, 1] 到 [0, 255]
rgb_np = (rgb_np * 255).astype(np.uint8)
# 创建一个PIL图像对象
print(rgb_np.shape)
image = Image.fromarray(rgb_np.transpose(1, 2, 0))  # 调整通道顺序
# 保存图像到本地文件
image.save("output4.png")