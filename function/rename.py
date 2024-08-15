import os

def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件列表
    files = os.listdir(folder_path)
    
    # 只保留.png格式的文件
    png_files = [file for file in files if file.endswith('.png')]
    
    # 对文件进行排序，确保按照某种顺序重命名
    png_files.sort()
    
    # 逐一重命名文件
    for idx, filename in enumerate(png_files):
        new_name = f"img{idx+1:06}.png"  # 生成新的文件名，类似 img000001.png
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')

# 使用示例
folder_path = r"/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu/DIV2K"   # 替换为你的文件夹路径
rename_images_in_folder(folder_path)
