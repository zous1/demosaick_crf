import cv2
import numpy as np
import os


path = r"F:/LuYD/Swin-Transformer-main/NeWCRFs-master/datasets/nyu/test_large"  
listdir = os.listdir(path)

# 新建split文件夹用于保存
pic_target = 'test_cut' # 分割后的图片保存的文件夹
pic_target = os.path.join(r"F:/LuYD/Swin-Transformer-main/NeWCRFs-master/datasets/nyu", pic_target)
if (os.path.exists(pic_target) == False):
        os.mkdir(pic_target)
# if not os.path.exists(pic_target):  #判断是否存在文件夹如果不存在则创建为文件夹
#     os.makedirs(pic_target)
#要分割后的尺寸

m = 0
# for循环迭代生成
for i in listdir:
    pic_path = os.path.join(path, i) # 分割的图片的位置
    # print(pic_path)
    # 读取要分割的图片，以及其尺寸等数据
    picture = cv2.imread(pic_path)
    (width, length, depth) = picture.shape
        
    m += 1

    if (width % 32 != 0 and length % 32 == 0) :
        pic = np.zeros((width - width % 32, length, depth))
        pic = picture[ 0 : width - width % 32, 0 : length, :]
    if (length % 32 != 0 and width % 32 == 0) :
        pic = np.zeros((width, length - length % 32, depth))
        pic = picture[ 0 : width, 0 : length - length % 32, :]
    if (width % 32 != 0 and length % 32 != 0) :
        pic = np.zeros((width - width % 32, length - length % 32, depth))
        pic = picture[ 0 : width - width % 32, 0 : length - length % 32, :]
        
    result_path = pic_target +"/"+ 'img{:d}.png'.format(m)
    print(result_path)
    cv2.imwrite(result_path, pic)

print("done!!!")