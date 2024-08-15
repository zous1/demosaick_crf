from PIL import Image

def resize_image(input_image_path, output_image_path, new_width, new_height):
    # 打开图像
    input_image = Image.open(input_image_path)
    
    # 调整大小
    output_image = input_image.resize((new_width, new_height))
    
    # 保存修改后的图像
    output_image.save(output_image_path)

# 输入和输出图像的路径
input_path = r"F:/LuYD/Swin-Transformer-main/NeWCRFs-master/datasets/nyu/test_large/0010.png"
output_path = "output_image.png"  # 替换成你的输出图像路径

# 调整大小为2040x1360
new_width = 2016
new_height = 1344

# 调用函数进行图像大小调整
resize_image(input_path, output_path, new_width, new_height)
