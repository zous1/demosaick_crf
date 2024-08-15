import cv2
import numpy as np
import os

path = r"/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu/train_large"  
listdir = os.listdir(path)

pic_target = 'train_cut256' 
pic_target = os.path.join(r"/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu", pic_target)
if not os.path.exists(pic_target):
    os.mkdir(pic_target)


cut_width = 256
cut_length = 256
m = 0

# for循环迭代生成
for i in listdir:
	pic_path = os.path.join(path, i) # 分割的图片的位置
	# print(pic_path)
	# 读取要分割的图片，以及其尺寸等数据
	picture = cv2.imread(pic_path)
	(width, length, depth) = picture.shape
	# 预处理生成0矩阵
	# 计算可以划分的横纵的个数
	num_width = int(( width - cut_width ) / 200 ) + 1
	num_length = int((length - cut_length) / 200 ) + 1
	for z in range(0, num_width):
		for j in range(0, num_length):
			m += 1
			pic = np.zeros((cut_width, cut_length, depth))
			pic = picture[z * 200 : ( z * 200 + cut_width ) , j * 200 : ( j * 200 + cut_length ), :]     
			result_path = pic_target +"/"+ 'img{:06d}.png'.format(m)
			print(result_path)
			cv2.imwrite(result_path, pic)
	if ( (num_width-1) * 200 + 256 ) < width :
		for j in range(0, num_length):
			m += 1
			pic = picture[width - 256 : width , j * 200 : ( j * 200 + cut_length ), :]
			result_path = pic_target +"/"+ 'img{:06d}.png'.format(m)
			print(result_path)
			cv2.imwrite(result_path, pic)
	if ( (num_length-1) * 200 + 256) < length :
		for i in range(0, num_width):
			m += 1
			pic = picture[i * 200 : ( i * 200 + cut_width ) , length - 256 : length, :]
			result_path = pic_target +"/"+ 'img{:06d}.png'.format(m)
			print(result_path)
			cv2.imwrite(result_path, pic)
	if ( (num_length-1) * 200 + 256) != length :
		if ( (num_width-1) * 200 + 256 ) != width :
			m += 1
			pic = picture[width - cut_width : width , length - cut_length : length, :]
			result_path = pic_target +"/"+ 'img{:06d}.png'.format(m)
			print(result_path)
			cv2.imwrite(result_path, pic)
	
print("done!!!")