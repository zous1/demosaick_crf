import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel//2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, kernel_size=5) if filter else nn.Identity()

        self.localap = Patch_ap(in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2)


    def forward(self, x):
        out = self.conv1(x)
       
        if self.filter:

            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
            
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, num_res):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock(in_channel, out_channel))
        self.layers.append(ResBlock(in_channel, out_channel, filter=True))
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.num_res = num_res

        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()

        for i, layer in enumerate(self.layers):
            if i == self.num_res//4:
                skip = x
                x = self.down(x)
            if i == self.num_res - self.num_res//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.conv(torch.cat((x, skip), dim=1))
            x = layer(x)

        return x + res

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.modulate = SFconv(inchannels)

    def forward(self, x):
        identity_input = x # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out

class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Conv2d(features, features, 1, 1, 0)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        
        out = self.out(fea_high + fea_low) 
        return out

class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        self.ap = nn.AdaptiveAvgPool2d((1,1))

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out