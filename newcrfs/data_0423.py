import os
import os.path as path
from os import listdir
import glob
import typing
import cv2
# from skimage.io import imsave, imread
import functools
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def is_image_file(filename):  # 找到以[".png", ".jpg", ".jpeg"]结尾的文件
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def modcrop(img, scale):  # 在CNN中，输入图像通常需要裁剪到特定的尺寸，否则将导致上采样或下采样时的边缘像素数据丢失
    # img: numpy, HWC or HW
    img = np.copy(img)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale  # 比如H=270，scale=4，H_r=2
        img = img[:H - H_r, :W - W_r]  # img[0:270-2, 0:270-2]
    elif img.ndim == 3:
        H, W, _ = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def shave(img, border=0):
    # img: numpy, HWC or HW
    img = np.copy(img)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img
    
def aug_data(img, rot_flip):  # 图像增强
    # img: numpy, HWC
    assert 0 <= rot_flip <= 7
    rot_flag = rot_flip // 2
    flip_flag = rot_flip % 2

    if rot_flag > 0:
        img = np.rot90(img, k=rot_flag, axes=(0, 1))  # 旋转90°
    if flip_flag >= 0:
        img = np.flip(img, axis=0)  # 翻转
    return img

class DataGenerator_1(Dataset):  # 就只是准备好图像的函数
    def __init__(self, data_dir='',
                patch_size = None,  # patch
                data_aug = False,  # 数据增强
                crop = True  # 裁剪
                ):
        # 1. Initialize file paths or a list of file names.
        super(DataGenerator_1, self).__init__() 

        assert path.isdir(data_dir)  # 断言一定有data_dir这个文件夹

        self.data_dir =  data_dir
        self.patch_size = patch_size
        self.data_augment = data_aug
        self.crop = crop

        imagefilenames = glob.glob(path.join(self.data_dir, 'train_cut384', '*'))  # 71-74行：获得按顺序的图像名称
        imagefilenames = [path.split(x)[1] for x in imagefilenames if is_image_file(x)]
        imagefilenames.sort()
        self.imagefilenames = imagefilenames
        # imagefilenames=['000001.png', '000002.png', '000003.png', '000004.png', '000005.png', '000006.png', ...]
            
    def __getitem__(self, index):
        file_index = self.imagefilenames[index]
        gt_data = self._read_png(file_index, 'train_cut384')  # 读入gt，gt是RGB图像
        gt_data = cv2.cvtColor(gt_data, cv2.COLOR_BGR2RGB)  
        # # normalization
        gt_data = gt_data.astype(np.float64) / 255.
        
        rgbw_data = self._read_png(file_index, 'train_cut384')  # 读入图像，这里是单通道的RGBW图像

        # 制作W_gt
        height, width, _ = gt_data.shape
        
        image = gt_data
        gt_w = np.zeros((height, width, 1))
        gt_w[:, :, 0] = image[:, :, 0] + image[:, :, 1] + image[:, :, 2]
        gt_w = gt_w / 3.

        # （2）获得有缺失值的W
        W_all = np.zeros((height, width))
        W_all[::2, ::2] = gt_w[::2, ::2, 0]
        W_all[1::2, 1::2] = gt_w[1::2, 1::2, 0]
        
        # （3）
        kodak_cfa = np.zeros((height, width, 3))
        # 赋值R
        kodak_cfa[3::4, 2::4, 0] = 1
        kodak_cfa[2::4, 3::4, 0] = 1
        
        # 赋值G   
        kodak_cfa[::4, 3::4, 1] = 1
        kodak_cfa[1::4, 2::4, 1] = 1
        kodak_cfa[2::4, 1::4, 1] = 1
        kodak_cfa[3::4, ::4, 1] = 1
        
        # 赋值B
        kodak_cfa[1::4, ::4, 2] = 1
        kodak_cfa[::4, 1::4, 2] = 1
#################################################删掉了与cfa相乘
        rgb_inRGBW = image 
                

        # （3）获得mask
        W_mask = np.zeros((height, width))
        W_mask[::2, ::2] = 1
        W_mask[1::2, 1::2] = 1

        # 
        W_all = np.expand_dims(W_all, axis=2)
        W_mask = np.expand_dims(W_mask, axis=2)

        # to tensor
        gt_data = ToTensor()(gt_data).to(torch.float32) # Converts a numpy.ndarray (H x W x C) [0, 255] to a torch.FloatTensor of shape (C x H x W) [0.0, 1.0]
        # rgbw_data = ToTensor()(rgbw_data).to(torch.float32)
        rgb_inRGBW = ToTensor()(rgb_inRGBW).to(torch.float32)
        gt_w = ToTensor()(gt_w).to(torch.float32)
        kodak_cfa = ToTensor()(kodak_cfa).to(torch.float32)

        W_all = ToTensor()(W_all).to(torch.float32)
        W_mask = ToTensor()(W_mask).to(torch.float32)

        # 3. Return a data pair 
        #####################################################################
        return rgb_inRGBW, gt_data, gt_w, W_all, W_mask, kodak_cfa
        # return gt_data
    
        # 

    def __len__(self):
        return len(self.imagefilenames) 

    @functools.lru_cache(maxsize=128)
    def _read_png(self, file_index, dtype):  # 读入图像
        if dtype == 'train_cut384':
            data = cv2.imread(self._png_path(file_index, dtype))
        elif dtype == 'train_cut384':
            data = cv2.imread(self._png_path(file_index, dtype), 0)
        # data = imread(self._png_path(file_index, dtype))  # 这里改读入方式
        return data

    def _png_path(self, file_index, dtype):  # 返回图像的地址
        if dtype == 'train_cut384':
            return path.join(self.data_dir, 'train_cut384', file_index)
        elif dtype == 'train_cut384':
            return path.join(self.data_dir, 'train_cut384', file_index)



if __name__=='__main__':
    ds = DataGenerator_1(patch_size=64)
    print(len(ds))
    for i in range(0, 5):
        data = ds[i]
        print(data[0].shape, data[0])
        print(data[1].shape, data[1])
        print(data[2].shape, data[2])
