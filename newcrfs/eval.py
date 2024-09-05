import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import os, sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from utils import post_process_depth, flip_lr, compute_errors
from networks.NewCRFDepth0420 import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='newcrfs')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='small07')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--noise',                                 help='if set, add gussian noise', action='store_true')
parser.add_argument('--noise_level',               type=float, help='if set, perform online eval in every eval_freq steps', default=0)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def eval(model, dataloader_eval, post_process=True):
    # eval_measures = torch.zeros(10).cuda()
    psnr_values = []  # 用于保存 PSNR 值的列表

    for i, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        # ... (之前的代码不变)
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            # print(gt_depth.shape)
            image_gt=image
            image = add_gaussian_noise(image,mean=0,std=args.noise_level) 
            pred_image = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_image = post_process_depth(pred_image, pred_depth_flipped)
        #去掉批量维度
        pred_image = pred_image.squeeze(0)  # 现在形状为 (3, 2016, 1344)
        
        # 假设 pred_image 是一个 PyTorch 张量
        # 先将其从 GPU 转移到 CPU（如果在 GPU 上），并转换为 NumPy 数组
        pred_image_np = pred_image.cpu().detach().numpy()
        
        # 如果 pred_image 是 3D 张量（如 [C, H, W]），则需要转置为 [H, W, C]
        if pred_image_np.shape[0] == 3:  # 检查是否为 3 通道图像
            pred_image_np = np.transpose(pred_image_np, (1, 2, 0))

        # 将 NumPy 数组转换为 PIL Image 对象
        pred_image_pil = Image.fromarray((pred_image_np *3 * 255).astype(np.uint8))

        file_path = f'datasets/nyu/output/pred_{i:02d}.png'
        pred_image_pil.save(file_path)  # 保存为 PNG 格式

        image1 = load_image(file_path)
        
        # 计算 PSNR 并添加到列表中
        psnr_value = compute_psnr(image_gt, image1)
        psnr_values.append(psnr_value)
        # pred_image=pred_image*255.0

        # # 将预测图像保存到文件中
        # pred_array = pred_image.squeeze(0).cpu().numpy()
        # pred_pil_image = Image.fromarray(pred_array.transpose(1, 2, 0).astype(np.uint8))
        # pred_pil_image = pred_pil_image.convert('RGB')
        # file_path = f'datasets/nyu/output/pred_{i}.png'
        # pred_pil_image.save(file_path)
        
        
    # 返回 PSNR 值列表
    return psnr_values


def compute_psnr(img1, img2, max_val=255.0):
    # img1_cpu = img1.cpu().numpy()*255.0
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    img1_cpu = img1*255
    img1_uint8 = np.uint8(img1_cpu)
    # img2_cpu = img2.cpu().numpy()*255.0
    img2_cpu = img2*255
    img2_uint8 = np.uint8(img2_cpu)
    mse = np.mean((img1_uint8 - img2_uint8) ** 2)
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr
    # pred = torch.from_numpy(pred)
    # gt = torch.from_numpy(gt)
    # mse = torch.mean((gt - pred) ** 2)
    # psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    # return psnr

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

def load_image(file_path):
    # 读取图像
    image = Image.open(file_path)
    # 转换为张量
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).cuda()
    #####读进来就是0-1的值
    return image_tensor

def main_worker(args):

    # CRF model
    ############################pretrained=None
    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
    #     eval_measures = eval(model, dataloader_eval, post_process=True)
    # model.eval()
    # with torch.no_grad():
        psnr_values = eval(model, dataloader_eval, post_process=True)

    # 打印 PSNR 值列表
    print("PSNR 值列表：", psnr_values)
    # 将张量转换为浮点数并计算均值
    psnr_values = [psnr.item() for psnr in psnr_values]
    mean_psnr = sum(psnr_values) / len(psnr_values)

    print("PSNR 值的均值:", mean_psnr)

def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
