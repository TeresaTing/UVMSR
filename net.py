import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from pscan import pscan

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class _MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        
        # old
        # self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
        #                       kernel_size=config.d_conv, bias=config.conv_bias, 
        #                       groups=config.d_inner,
        #                       padding=config.d_conv - 1)

        self.forward_conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        self.backward_conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        self.norm = RMSNorm(config.d_model)

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        skip = x
        x = self.norm(x)

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x1 branch
        x = x.transpose(1, 2) # (B, ED, L)
        forward_conv_output = self.forward_conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        forward_conv_output = forward_conv_output.transpose(1, 2) # (B, L, ED)
        x1_ssm = self.ssm(forward_conv_output)

        # x2 branch
        backward_conv_output = self.backward_conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        backward_conv_output = backward_conv_output.transpose(1, 2) # (B, L, ED)
        x2_ssm = self.ssm(backward_conv_output)

        # z branch
        # z = F.silu(z)
        z = F.mish(z)

        x1 = x1_ssm * z
        x2 = x2_ssm * z
        x = x1 + x2

        output = self.out_proj(x)
        output = output + skip

        return output
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()
        # T ODO remove .float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class PatchEmbed(nn.Module):
    def __init__(self,in_feature,out_feature,final_feature,act=nn.LeakyReLU(),dropout=0,is_BN=False,flatten=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.final_feature = final_feature
        self.flatten = flatten

        m = []
        m.append(nn.Conv2d(self.in_feature, self.out_feature, 1, 1, 0))
        if is_BN:
            m.append(nn.BatchNorm2d(self.out_feature))
        m.append(act)
        if dropout>0:
            m.append(nn.Dropout(dropout))

        m.append(nn.Conv2d(self.out_feature, self.out_feature, 3, 1, 1))
        if is_BN:
            m.append(nn.BatchNorm2d(self.out_feature))
        m.append(act)
        if dropout>0:
            m.append(nn.Dropout(dropout))

        m.append(nn.Conv2d(self.out_feature, self.final_feature, 1, 1, 0))

        self.this_block = nn.Sequential(*m)
    def forward(self, x):
        res = self.this_block(x)
        # if self.in_feature==self.final_feature:
        #     res += x
        if self.flatten:
            res = res.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return res

class Downsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelUnshuffle(2))
                m.append(nn.Conv2d(4 * num_feat, num_feat, 3, 1, 1))
        elif scale == 3:   
            m.append(nn.PixelUnshuffle(3))
            m.append(nn.Conv2d(9 * num_feat, num_feat, 3, 1, 1))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Downsample, self).__init__(*m)

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



class Conv_Block(nn.Module):
    def __init__(self,in_feature,out_feature,final_feature,act=nn.LeakyReLU(),dropout=0,is_BN=False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.final_feature = final_feature
        m = []

        m.append(nn.Conv2d(self.in_feature, self.out_feature, 1, 1, 0))
        if is_BN:
            m.append(nn.BatchNorm2d(self.out_feature))
        m.append(act)
        if dropout>0:
            m.append(nn.Dropout(dropout))

        m.append(nn.Conv2d(self.out_feature, self.out_feature, 3, 1, 1))
        if is_BN:
            m.append(nn.BatchNorm2d(self.out_feature))
        m.append(act)
        if dropout>0:
            m.append(nn.Dropout(dropout))

        m.append(nn.Conv2d(self.out_feature, self.final_feature, 1, 1, 0))

        self.this_block = nn.Sequential(*m)
    def forward(self, x):
        res = self.this_block(x)
        if self.in_feature==self.final_feature:
            res += x
        return res



class UVMSR(nn.Module):
    def __init__(self,n_color):
        super(UVMSR, self).__init__()
        self.n_color = n_color
        self.n_feats = 2 * self.n_color
        self.img_size = 128

        self.config=MambaConfig(self.n_feats)

        act = nn.LeakyReLU()
       

        self.first_conv = Conv_Block(self.n_color,self.n_feats,self.n_feats,act,0)

        # x1 L
        self.embed_x1_l = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x1_l = _MambaBlock(self.config)
        self.conv_x1_l = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)

        self.stride_down_x1 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats,2,2))


        # x2 L
        self.embed_x2_l = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x2_l = _MambaBlock(self.config)
        self.conv_x2_l = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.stride_down_x2 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats,2,2))


        # x4 L
        self.embed_x4_l = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x4_l = _MambaBlock(self.config)
        self.conv_x4_l = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.stride_down_x4 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats,2,2))

        # x8 L
        self.embed_x8_l = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x8_l = _MambaBlock(self.config)
        self.conv_x8_l = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.stride_down_x8 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats,2,2))

        # x16 M
        self.embed_x16 = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x16 = _MambaBlock(self.config)
        self.conv_x16 = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.up_x16 = Upsample(2,self.n_feats)

        # x8 R
        self.embed_x8_r = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x8_r = _MambaBlock(self.config)
        self.conv_x8_r = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.up_x8 = Upsample(2,self.n_feats)

        # x4 R
        self.embed_x4_r = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x4_r = _MambaBlock(self.config)
        self.conv_x4_r = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.up_x4 = Upsample(2,self.n_feats)

        # x2 R
        self.embed_x2_r = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x2_r = _MambaBlock(self.config)
        self.conv_x2_r = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)
        self.up_x2 = Upsample(2,self.n_feats)

        # x1 R
        self.embed_x1_r = PatchEmbed(self.n_feats,self.n_feats,self.n_feats,act,0)
        self.mamba_x1_r = _MambaBlock(self.config)
        self.conv_x1_r = Conv_Block(self.n_feats,self.n_feats,self.n_feats,act,0.2)

        self.final_conv = nn.Sequential(nn.Conv2d(self.n_feats, self.n_color, 1, 1, 0))


    def forward(self, x: torch.Tensor):
        input_x = x
        b,c,h,w = x.shape
        x = self.first_conv(x)
        # x1 L
        x1_l_in = x
        x = self.embed_x1_l(x)
        x = self.mamba_x1_l(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size,self.img_size)
        x = self.conv_x1_l(x)
        x1_l_out = x1_l_in + x
        # print('x1_l_out shape : ' + str(x1_l_out.shape))
        x = self.stride_down_x1(x1_l_out)

        # x2 L
        x2_l_in = x
        x = self.embed_x2_l(x)
        x = self.mamba_x2_l(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//2,self.img_size//2)
        x = self.conv_x2_l(x)
        x2_l_out = x2_l_in + x
        # print('x2_l_out shape : ' + str(x2_l_out.shape))
        x = self.stride_down_x2(x2_l_out)

        # x4 L
        x4_l_in = x
        x = self.embed_x4_l(x)
        x = self.mamba_x4_l(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//4,self.img_size//4)
        x = self.conv_x4_l(x)
        x4_l_out = x4_l_in + x
        # print('x4_l_out shape : ' + str(x4_l_out.shape))
        x = self.stride_down_x4(x4_l_out)    

        # x8 L
        x8_l_in = x
        x = self.embed_x8_l(x)
        x = self.mamba_x8_l(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//8,self.img_size//8)
        x = self.conv_x8_l(x)
        x8_l_out = x8_l_in + x
        # print('x8_l_out shape : ' + str(x8_l_out.shape))
        x = self.stride_down_x8(x8_l_out)  


        # x16 M
        x16_in = x
        x = self.embed_x16(x)
        x = self.mamba_x16(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//16,self.img_size//16)
        x = self.conv_x16(x)
        x = x16_in + x
        x = self.up_x16(x)
        
        # x8 R
        x8_r_in = x
        x = x + x8_l_out
        x = self.embed_x8_r(x)
        x = self.mamba_x8_r(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//8,self.img_size//8)
        x = self.conv_x8_r(x)
        x = x8_r_in + x
        # print(x.shape)
        x = self.up_x8(x)
        # print(x.shape)

        # x4 R
        x4_r_in = x
        x = x + x4_l_out
        x = self.embed_x4_r(x)
        x = self.mamba_x4_r(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//4,self.img_size//4)
        x = self.conv_x4_r(x)
        x = x4_r_in + x
        # print(x.shape)
        x = self.up_x4(x)
        # print(x.shape)

        # x2 R
        x2_r_in = x
        x = x + x2_l_out
        x = self.embed_x2_r(x)
        x = self.mamba_x2_r(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size//2,self.img_size//2)
        x = self.conv_x2_r(x)
        x = x2_r_in + x
        # print(x.shape)
        x = self.up_x2(x)
        # print(x.shape)

        # x1 R
        x1_r_in = x
        x = x + x1_l_out
        x = self.embed_x1_r(x)
        x = self.mamba_x1_r(x)
        x = x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.img_size,self.img_size)
        x = self.conv_x1_r(x)
        x = x1_r_in + x
        # print(x.shape)

        # x = x.reshape(b,1,2*c,h,w)
        x = self.final_conv(x)
        # x = x.reshape(b,c,h,w)
        # print(x.shape)
        x = input_x + x
        return x




