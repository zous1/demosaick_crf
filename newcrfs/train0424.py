import os
import argparse
import time
import math
import numpy as np
from data_0423 import DataGenerator_1 as DataGenerator
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks.NewCRFDepth0420 import NewCRFDepth as Net
from utils import  compute_psnr



# set flags / seeds
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for Depth SR")
    # parser.add_argument('--data_dir', type=str, default='./dataset/train_data')
    parser.add_argument('--data_dir', type=str, default='datasets/nyu')
    parser.add_argument('--batch_size', type=int, default=2)  # 原本20
    parser.add_argument('--patch_size', type=int, default=60)
    parser.add_argument('--data_augment', type=bool, default=False)
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=8)  # 原本8
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--snapshot_dir', type=str, default='models/newcrf_nyu/')
    parser.add_argument('--snapshot', type=int, default=1) 
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=500000)
    parser.add_argument('--prinf_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=1) 
    parser.add_argument('--val_data_dir', type=str, default='datasets/nyu/test_cut')
    parser.add_argument('--result_save_dir', type=str, default='datasets/nyu/output/')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--resume', type=bool, default=False)  # 判断是否读入权重
    parser.add_argument('--resume_dir', type=str, default='') 
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--end_learning_rate', type=float, default=-1)

    opt = parser.parse_args()
    print(opt)

   
    # if os.path.exists(opt.data_dir):
    #     print("数据集文件存在，路径正确。")
    # else:
    #     print("数据集文件不存在，请检查路径。")

    ###############################
    ## step1：W插值 ##
    # 改这里的地址，DataGenerator的地址里再增加一个插值后的W地址
    # add code for datasets
    print("===> Loading datasets")
    data_set = DataGenerator(data_dir= opt.data_dir,
                patch_size = opt.patch_size,
                data_aug = opt.data_augment,
                crop = opt.crop)
    print(len(data_set))
    val_data_set = DataGenerator(data_dir= opt.val_data_dir,
                                data_aug =False,
                                crop = False)
    training_data = DataLoader(dataset=data_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, drop_last=True)
    val_data = DataLoader(dataset=val_data_set, batch_size=1, num_workers=opt.workers, shuffle=False)
    
    # instantiate network
    print("===> Building model")
    devices_ids = list(range(opt.n_gpus))
    net = Net(version='small07', inv_depth=False)

    # if running on GPU and we want to use cuda move model there
    print("===> Setting GPU")
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    net = nn.DataParallel(net, device_ids=devices_ids)  # 使得网络能在GPU上并行运算
    net = net.cuda()

    # create loss
    criterion_L1_cb = nn.MSELoss()
    criterion_L1_cb = criterion_L1_cb.cuda()
    
    # print('---------- Networks architecture -------------')
    # print_network(net)
    # print('----------------------------------------------')

    # optionally ckp from a checkpoint
    if opt.resume:  # 读入权重，继续训练
        if opt.resume_dir != None:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_state_dict(torch.load(opt.resume_dir))
            else:
                net.load_state_dict(torch.load(opt.resume_dir))
            print('Net work loaded from {}'.format(opt.resume_dir))

    # create optimizer
    print("===> Setting Optimizer")
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr)

    print("===> Training")
    curr_epoch = 0  
    curr_step = 0
    ###########################学习率的衰减
    end_learning_rate = opt.end_learning_rate if opt.end_learning_rate != -1 else 0.1 * opt.lr
    steps_per_epoch = training_data.__len__()
    num_total_steps = opt.n_epochs * steps_per_epoch

    while curr_epoch < opt.n_epochs:
        net.train()
        mean_loss = 0
        
        # learning rate is decayed with poly policy 学习率随着epoch变化
        lr_ = opt.lr * (1 - float(curr_step)/opt.n_iters)**2      
        
        with tqdm(total=training_data.__len__(), desc="training process") as tq:

            for _, batch in enumerate(training_data, 1):
                optim.zero_grad()
                curr_step += 1
                rgb_inRGBW, gt_data, gt_w, W_all, W_mask, rgb_mask = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3]), Variable(batch[4], requires_grad=False),  Variable(batch[5], requires_grad=False)
                rgb_inRGBW = rgb_inRGBW.cuda()
                gt_data = gt_data.cuda()
                w_gt = gt_w.cuda()
                W_all = W_all.cuda()
                W_mask = W_mask.cuda() 
                rgb_mask = rgb_mask.cuda()

                t0 = time.time()
                

               
                img_pred, w_output = net(rgb_inRGBW)
                # print(img_pred)
                loss = criterion_L1_cb(img_pred * 3., gt_data)

                t1 = time.time()

                loss.backward()
                for param_group in optim.param_groups:
                    param_group['lr'] = lr_
                optim.step()
                
                mean_loss += loss.item()

                tq.set_postfix({"loss": "{0:1.5f}".format(loss.cpu().detach().numpy().item()), })
                tq.update(1)

        
        print("===> Loss: epoch [{}] || lr={:.6f} || loss: {}".format(curr_epoch, lr_, mean_loss/len(training_data)))
        # print()

        if curr_epoch % opt.val_freq == 0:
            print('Evaluation....')
            net.eval()
            mean_rmse = 0
            mean_psnr = 0
            mean_rmse_w = 0
            mean_psnr_w = 0
            n_count, n_total = 1, len(val_data)
            
            with tqdm(total=val_data.__len__(), desc="valing process") as tq:
            
                for batch in val_data:

                    rgbw_data, gt_data, gt_w, W_all, W_mask, rgb_mask = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3]), Variable(batch[4], requires_grad=False),  Variable(batch[5], requires_grad=False)

                    rgbw_data = rgbw_data.cuda()
                    gt_data = gt_data.cuda()
                    w_gt = gt_w.cuda()
                    W_all = W_all.cuda()
                    W_mask = W_mask.cuda()
                    rgb_mask = rgb_mask.cuda()

                    with torch.no_grad():
                        img_pred, w_output = net(rgbw_data, W_all, W_mask, rgb_mask)

                    psnr = compute_psnr(img_pred * 3., gt_data)
                    mean_psnr += psnr

                    psnr_w = compute_psnr(w_output, w_gt)
                    mean_psnr_w += psnr_w

                    image_name = val_data_set.imagefilenames[n_count-1]

                    img_pred = img_pred.permute(0, 2, 3, 1)
                    # save_img(img_pred.cpu(), image_name, opt.result_save_dir, 'rgb')
                    n_count += 1
                    
                    tq.set_postfix({"n_count": "{0:1.6f}".format(psnr), })
                    tq.update(1)
                    
            mean_rmse /= len(val_data)
            mean_psnr /= len(val_data)
            mean_rmse_w /= len(val_data)
            mean_psnr_w /= len(val_data)

            print("Valid  iter [{}] || rmse: {} || psnr: {}".format(curr_epoch, mean_rmse, mean_psnr))
            print("w || rmse: {} || psnr: {}".format(mean_rmse_w, mean_psnr_w))
            print()

        # if curr_epoch % opt.snapshot == 0:
            # save_model(net, curr_epoch, opt.snapshot_dir)

        curr_epoch += 1
