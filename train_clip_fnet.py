#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
from PIL import Image
import torch
from torch.fft import fftn
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchtyping import TensorType, patch_typeguard
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm import trange
from tqdm.std import tqdm
from typing import List, Optional, Tuple
import wandb
import numpy as np
from torch.nn.parameter import Parameter
from functools import reduce
from functools import partial

from CLIP import clip


p = argparse.ArgumentParser()
p.add_argument(
    '--train-set',
    type=Path,
    required=True,
    help='the training set location')
""" p.add_argument(
    '--demo-prompts',
    type=Path,
    required=True,
    help='the demo prompts'
) """
p.add_argument('--batch-size', type=int, default=2)
p.add_argument('--run-name', type=str)
p.add_argument('--seed', type=int, default=21)
p.add_argument('--grad-accum', type=int, default=8)


# Define utility functions


def stat_cuda(msg):
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, reserved: %dM, max reserved: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024,
        torch.cuda.max_memory_reserved() / 1024 / 1024
    ))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = SpectralConv2d(3, 32, modes=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, modes=3)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=1, modes=3)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=1, modes=3)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=1, modes=3)
        self.linear1 = nn.Linear(32*64*block.expansion, num_classes)
        # self.linear2 = nn.Linear(100, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, modes=10):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, modes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer2(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        # out = F.relu(out)
        # out = self.linear2(out)
        return out


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param.to('cpu'), alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the diffusion noise schedule

def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([
            nn.Linear(f_in, f_mid),
            nn.ReLU(inplace=True),
            nn.Linear(f_mid, f_out),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class Modulation2d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.InstanceNorm2d(c_mid, affine=False),
            Modulation2d(state, feats_in, c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.InstanceNorm2d(c_out, affine=False) if not is_last else nn.Identity(),
            Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)



class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)
        # self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.InstanceNorm2d(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input: TensorType['batch', -1, -1, -1]):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        c = 128  # The base channel count
        cs = [c, c * 2, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(512 + 128, 1024, 1024),
            ResLinearBlock(1024, 1024, 1024, is_last=True),
        )

        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5**0.5

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(   # 256x256
            conv_block(3 + 16, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SpectralConv2d(cs[0], cs[0], 128, 128),
            #conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 128x128
                conv_block(cs[0], cs[1], cs[1]),
                #conv_block(cs[1], cs[1], cs[1]),
                SpectralConv2d(cs[1], cs[1], 64, 64),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 64x64
                    conv_block(cs[1], cs[2], cs[2]),
                    #conv_block(cs[2], cs[2], cs[2]),
                    SpectralConv2d(cs[2], cs[2], 64, 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,  # 32x32
                        conv_block(cs[2], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,  # 16x16
                            conv_block(cs[3], cs[4], cs[4]),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            conv_block(cs[4], cs[4], cs[4]),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            conv_block(cs[4], cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            conv_block(cs[4], cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            SpectralConv2d(cs[4], cs[4] // 64, 8, 8),
                            SkipBlock([
                                self.down,  # 8x8
                                conv_block(cs[4], cs[5], cs[5]),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                SpectralConv2d(cs[5], cs[5] // 64, 4, 4),
                                conv_block(cs[5], cs[5], cs[5]),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                SpectralConv2d(cs[5], cs[5] // 64, 4, 4),
                                conv_block(cs[5], cs[5], cs[5]),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SpectralConv2d(cs[5], cs[5], 4, 4),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                SkipBlock([
                                    self.down,  # 4x4
                                    conv_block(cs[5], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    #SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    conv_block(cs[6], cs[6], cs[5]),
                                    # SelfAttention2d(cs[5], cs[5] // 64),
                                    SpectralConv2d(cs[6], cs[6], 2, 2),
                                    self.up,
                                ]),
                                conv_block(cs[5] * 2, cs[5], cs[5]),
                                SpectralConv2d(cs[5], cs[5], 4, 4),
                                #SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SpectralConv2d(cs[5], cs[5], 4, 4),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SpectralConv2d(cs[5], cs[5], 4, 4),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[4]),
                                SpectralConv2d(cs[5], cs[5], 4, 4),
                                # SelfAttention2d(cs[4], cs[4] // 64),
                                self.up,
                            ]),
                            conv_block(cs[4] * 2, cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            conv_block(cs[4], cs[4], cs[4]),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[3]),
                            SpectralConv2d(cs[4], cs[4], 8, 8),
                            # SelfAttention2d(cs[3], cs[3] // 64),
                            self.up,
                        ]),
                        conv_block(cs[3] * 2, cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[2]),
                        self.up,
                    ]),
                    conv_block(cs[2] * 2, cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[1]),
                    self.up,
                ]),
                conv_block(cs[1] * 2, cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[0]),
                self.up,
            ]),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 3, is_last=True),
        )

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5**0.5

    def forward(self, input, t, clip_embed):
        clip_embed = F.normalize(clip_embed, dim=-1) * clip_embed.shape[-1]**0.5
        mapping_timestep_embed = self.mapping_timestep_embed(t[:, None])
        self.state['cond'] = self.mapping(torch.cat([clip_embed, mapping_timestep_embed], dim=1))
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out


@torch.no_grad()
def sample(model, x, steps, eta, extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


class TokenizerWrapper:
    def __init__(self, max_len=None):
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.context_length = 77
        self.max_len = self.context_length - 2 if max_len is None else max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros([len(texts), self.context_length], dtype=torch.long)
        for i, text in enumerate(texts):
            tokens_trunc = self.tokenizer.encode(text)[:self.max_len]
            tokens = [self.sot_token, *tokens_trunc, self.eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result


class ImageDataset(data.Dataset):
    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""
    
    def __init__(self, folder, preprocess_im, preprocess_text, enable_text=True, enable_image=True):
        super().__init__()
        path = Path(folder)
        self.enable_text = enable_text
        self.enable_image = enable_image
        if self.enable_text:
            text_files = [*path.glob("**/*.txt")]
            text_files = {text_file.stem: text_file for text_file in text_files}
            if len(text_files) == 0:
                self.enable_text = False
        if self.enable_image:
            image_files = [
                *path.glob("**/*.png"),
                *path.glob("**/*.jpg"),
                *path.glob("**/*.jpeg"),
                *path.glob("**/*.bmp"),
            ]
            image_files = {image_file.stem: image_file for image_file in image_files}
            if len(image_files) == 0:
                self.enable_image = False
        keys = None
        join = lambda new_set: new_set & keys if keys is not None else new_set
        if self.enable_text:
            keys = join(text_files.keys())
        elif self.enable_image:
            keys = join(image_files.keys())

        self.keys = list(keys)
        if self.enable_text:
            self.text_files = {k: v for k, v in text_files.items() if k in keys}
            self.text_transform = preprocess_text
        if self.enable_image:
            self.image_files = {k: v for k, v in image_files.items() if k in keys}
            self.image_transform = preprocess_im

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        try:
            if self.enable_image:
                image_file = self.image_files[key]
                image_tensor = self.image_transform(Image.open(image_file))
            if self.enable_text:
                text_file = self.text_files[key]
                caption = text_file.read_text()
                text = self.text_transform(caption)
        except (Image.UnidentifiedImageError, OSError, KeyError, Image.DecompressionBombError,):
            print(f"Failed to load image/text {key}. Skipping.")
            return None  # return None to be filtered in the batch collate_fn
        return image_tensor, text



class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


def make_cutouts(input, cut_size):
    cutn = 16
    cut_pow = 1
    sideY, sideX = input.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, cut_size)
    cutouts = []
    for _ in range(cutn):
        size = int(torch.rand([])**cut_pow * (max_size - min_size) + min_size)
        offsetx = torch.randint(0, sideX - size + 1, ())
        offsety = torch.randint(0, sideY - size + 1, ())
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutout = F.adaptive_avg_pool2d(cutout, cut_size)
        cutouts.append(cutout)
    return torch.cat(cutouts)


def clip_image_embed(clip_model, images):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    imgs = TF.resize(
        images,
        clip_model.visual.input_resolution,
        transforms.InterpolationMode.BICUBIC
    )
    embeds = F.normalize(clip_model.encode_image(normalize(imgs)).float(), dim=1)
    return embeds

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def eval_step(model, clip_model, batch, ):
    images, captions = batch
    images = images.to('cuda')
    # Sample timesteps
    t = torch.rand(images.shape[0]).to('cuda')
    # Calculate the noise schedule parameters for those timesteps
    alphas, sigmas = get_alphas_sigmas(t)
    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]
    noise = torch.randn_like(images)
    noised_images = images * alphas + noise * sigmas
    with torch.no_grad():
        cond = clip_model.encode_text(captions.to('cuda'))
    grads = -torch.autograd.functional.vjp(
        lambda x: spherical_dist_loss(cond, clip_image_embed(clip_model, x)),
        noised_images,
        torch.ones(noised_images.shape[0], device='cuda')
    )[1]
    clip_model.zero_grad(set_to_none=True)
    targets = noise * alphas - grads * sigmas
    # Compute the model output and the loss.
    # with torch.cuda.amp.autocast():
    pred = model(noised_images, t, cond)
    loss = F.mse_loss(pred, targets)
    return loss


def train_step(model, clip_model, batch):
    loss = eval_step(model, clip_model, batch)
    log_dict = {'train/loss': loss}
    wandb.log(log_dict)
    return loss


def get_models():
    diffusion_model = DiffusionModel()
    model_ema = deepcopy(diffusion_model)
    clip_model = clip.load('ViT-B/16', 'cpu', jit=False)[0].eval()
    return diffusion_model, model_ema, clip_model

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)



def do_ema_upodate(model, model_ema, global_step):
    if global_step < 20000:
        decay = 0.99
    elif global_step < 200000:
        decay = 0.999
    else:
        decay = 0.9999
    ema_update(model, model_ema, decay)


@torch.no_grad()
def on_batch_end(prompts, prompt_tokens, trainer, model, clip_model):
    lines = [f'({i // 4}, {i % 4}) {line}' for i, line in enumerate(prompts)]
    lines_text = '\n'.join(lines)
    Path('demo_prompts_out.txt').write_text(lines_text)

    noise = torch.randn([16, 3, 256, 256], device=model.device)
    clip_embed = clip_model.encode_text(prompt_tokens.to(model.device))
    model.eval()
    fakes = sample(model, noise, 1000, 1, {'clip_embed': clip_embed})

    grid = utils.make_grid(fakes, 4, padding=0).cpu()
    image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
    filename = f'demo_{trainer.global_step:08}.png'
    image.save(filename)
    log_dict = {
        'demo_grid': wandb.Image(image),
        'prompts': wandb.Html(f'<pre>{lines_text}</pre>')
    }
    wandb.log(log_dict)
    model.train()



def get_dataloader(train_fp, size, batch_size):
    tok_wrap = TokenizerWrapper()

    def ttf(caption):
        return tok_wrap(caption).squeeze(0)
    tf = transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = ImageDataset(train_fp, tf, ttf)
    train_dl = data.DataLoader(
        train_set,
        batch_size,
        sampler=data.RandomSampler(train_set),
        num_workers=6,
        persistent_workers=True
    )
    return train_dl


def train_loop(model, ema, clip_model, optimizer, data_loader, accum_iter, init_step=0):
    model.train()
    # scaler = GradScaler()
    for i, batch in enumerate(tqdm(data_loader)):
        loss = train_step(model, clip_model, batch)
        loss /= accum_iter
        #scaler.scale(loss).backward()
        loss.backward()
        if ((i + 1) % accum_iter == 0) or ((i + 1) == len(data_loader)):
            optimizer.step()
            #scaler.step(optimizer)
            #scaler.update()
            optimizer.zero_grad()
        do_ema_upodate(model, ema, i + init_step)
        if (i + init_step) % 1000 == 0 and i > 0:
            torch.save(
                {
                    'model': model,
                    'ema': ema,
                    'optimizer': optimizer,
                    'rng_state': torch.random.get_rng_state() 
                },
                f'./checkpoints/clip_cond/ckpt_{i+init_step}.pt'
            )
    return i
    

def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args, _ = p.parse_known_args()
    batch_size = args.batch_size
    size = 256
    torch.manual_seed(args.seed)

    wandb.init(
        name=args.run_name,
        config={
            'batch_size': batch_size,
            "image_size": size,
            "dataset": args.train_set,
            'seed': args.seed
        }
    )
    data = get_dataloader(args.train_set, size, batch_size)

    # demo_prompts = [line.rstrip() for line in open(args.demo_prompts).readlines()]

    model, ema_model, clip_model = get_models()
    num_params = param_count(model)
    print(f'Diffusion model # of parameters: {num_params}')
    wandb.config.update({'num_params': num_params})
    opt = get_optimizer(model)
    wandb.watch(model)
    global_counter = 0
    model.to('cuda')
    clip_model.to('cuda')
    ema_model.to('cpu')
    for _ in range(100):
        global_counter += train_loop(
            model,
            ema_model,
            clip_model,
            opt,
            data,
            args.grad_accum,
            global_counter
        )
        


if __name__ == '__main__':
    main()
