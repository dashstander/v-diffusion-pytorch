#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
import sys

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm import tqdm, trange

from CLIP import clip


# Define utility functions


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class AdaGN(nn.Module):
    def __init__(self, state, feats_in, c_out, num_groups, eps=1e-5):
        super().__init__()
        self.state = state
        self.norm = nn.GroupNorm(num_groups, c_out, eps, affine=False)
        self.layer = nn.Linear(feats_in, c_out * 2)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        input = self.norm(input)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ChangeChannelCount(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        if c_in % c_out and c_out % c_in:
            raise RuntimeError('Channels in and channels out must be compatible.')
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, input):
        n, c, h, w = input.shape
        if self.c_in == self.c_out:
            return input
        if self.c_in > self.c_out:
            return input.view([n, self.c_in // self.c_out, self.c_out, h, w]).mean(1)
        return input.repeat([1, self.c_out // self.c_in, 1, 1])


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out):
        skip = None if c_in == c_out else ChangeChannelCount(c_in, c_out)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            AdaGN(state, feats_in, c_mid, 1),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            AdaGN(state, feats_in, c_out, 1),
            nn.GELU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main):
        super().__init__()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return (self.main(input) + input) / 2


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

        

class SelfAttention2d(nn.Module):
    def __init__(self, state, feats_in, c_in, n_head=1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = AdaGN(state, feats_in, c_in, 1)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2d(nn.Module):
    def __init__(self, state, feats_in, c_enc, c_dec, n_head=1):
        super().__init__()
        assert c_dec % n_head == 0
        self.state = state
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = AdaGN(state, feats_in, c_dec, 1)
        self.n_head = n_head
        self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Conv2d(c_dec, c_dec, 1)

    def forward(self, input):
        n, c, h, w = input.shape
        q = self.q_proj(self.norm_dec(input))
        q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
        kv = self.kv_proj(self.norm_enc(self.state['enc_hidden']))
        kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
        k, v = kv.chunk(2, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale))
        att = att - (self.state['enc_padding_mask'][:, None, None, :]) * 10000
        att = att.softmax(3)
        y = (att @ v).transpose(2, 3)
        y = y.contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class Downsample2d(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.tensor([[1, 3, 3, 1]]) / 8
        self.register_buffer('weight', (weight.T @ weight)[None, None])
    
    def forward(self, input):
        n, c, h, w = input.shape
        input = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input, self.weight.repeat([c, 1, 1, 1]), stride=2, groups=c)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        d_model = 512
        seq_len = 77

        encoder_layer = nn.TransformerEncoderLayer(d_model, d_model // 64, d_model * 4,
                                                   dropout=0, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 12)
        self.token_embed = nn.Embedding(49408, d_model)
        self.enc_pos_embed = nn.Parameter(torch.randn([seq_len, d_model]))
        self.enc_timestep_embed = FourierFeatures(1, d_model, std=10)
        self.register_buffer('null_text', clip.tokenize(''))

        # Decoder
        c = 96  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4]

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, d_model)
        self_block = partial(SelfAttention2d, self.state, d_model)
        cross_block = partial(CrossAttention2d, self.state, d_model, d_model)

        self.dec_timestep_embed = FourierFeatures(1, 32, std=10)
        self.dec_pos_embed = FourierFeatures2d(2, 16, std=10)

        self.down = Downsample2d()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.decoder = nn.Sequential(   # 64x64
            nn.Conv2d(3 * 2 + 32 + 16, cs[0], 1),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 32x32
                ChangeChannelCount(cs[0], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 16x16
                    ChangeChannelCount(cs[1], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    self_block(cs[2], cs[2] // 64),
                    cross_block(cs[2], cs[2] // 64),
                    self_block(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    self_block(cs[2], cs[2] // 64),
                    cross_block(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    cross_block(cs[2], cs[2] // 64),
                    SkipBlock([
                        self.down,  # 8x8
                        ChangeChannelCount(cs[2], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        SkipBlock([
                            self.down,  # 16x16
                            ChangeChannelCount(cs[3], cs[4]),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            self_block(cs[4], cs[4] // 64),
                            cross_block(cs[4], cs[4] // 64),
                            ChangeChannelCount(cs[4], cs[3]),
                            self.up,
                        ]),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        self_block(cs[3], cs[3] // 64),
                        cross_block(cs[3], cs[3] // 64),
                        ChangeChannelCount(cs[3], cs[2]),
                        self.up,
                    ]),
                    conv_block(cs[2], cs[2], cs[2]),
                    self_block(cs[2], cs[2] // 64),
                    cross_block(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    self_block(cs[2], cs[2] // 64),
                    cross_block(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    self_block(cs[2], cs[2] // 64),
                    cross_block(cs[2], cs[2] // 64),
                    ChangeChannelCount(cs[2], cs[1]),
                    self.up,
                ]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                ChangeChannelCount(cs[1], cs[0]),
                self.up,
            ]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            nn.Conv2d(cs[0], 3, 1),
        )

        with torch.no_grad():
            for p in self.decoder.parameters():
                p.mul_(1/4)

    def forward(self, input, t, text):
        # Encoder
        enc_timestep_embed = self.enc_timestep_embed(t[:, None])[:, None]
        token_embed = self.token_embed(text[:, 1:])
        enc_embed = torch.cat([enc_timestep_embed, token_embed], dim=1) + self.enc_pos_embed
        eot_mask = text == 49407
        enc_padding_mask = eot_mask.long().cumsum(1).eq(1) & ~eot_mask
        enc_hidden = self.encoder(enc_embed, src_key_padding_mask=enc_padding_mask)

        # Decoder
        dec_timestep_embed = expand_to_planes(self.dec_timestep_embed(t[:, None]), input.shape)
        pos_y = torch.linspace(-1, 1, input.shape[2], device=input.device)
        pos_x = torch.linspace(-1, 1, input.shape[3], device=input.device)
        grid = torch.stack(torch.meshgrid(pos_y, pos_x))[None].repeat([input.shape[0], 1, 1, 1])
        dec_pos_embed = self.dec_pos_embed(grid)
        self.state['cond'] = enc_hidden[eot_mask]
        self.state['enc_hidden'] = enc_hidden
        self.state['enc_padding_mask'] = enc_padding_mask
        dec_input = torch.cat([input, dec_timestep_embed, dec_pos_embed], dim=1)
        # dec_input = torch.cat([input, dec_timestep_embed], dim=1)
        out = self.decoder(dec_input)

        # Clean up
        self.state.clear()
        return out


@torch.no_grad()
def sample_sde(model, x, n_steps, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    end_step = 1e-3
    steps = torch.linspace(1, end_step, n_steps + 1)
    x, v = x.chunk(2, dim=1)
    v /= 4**0.5

    # The sampling loop
    for i in trange(len(steps) - 1):

        # Get the model output
        with torch.cuda.amp.autocast():
            eps = model(torch.cat([x, v], dim=1), ts * steps[i], **extra_args).float()

        t = steps[i]
        sigma = make_sigma_hsm(t[None])[0]
        sigma_xx, sigma_xv, sigma_vv = sigma[0, 0], sigma[0, 1], sigma[1, 1]
        l_t = torch.sqrt(sigma_xx / (sigma_xx * sigma_vv - sigma_xv**2))
        score = -eps * l_t

        dt = steps[i] - steps[i + 1]
        dx = -16 * v * dt
        dv = (x - 4 * v + 2 * (score + 4 * v)) * 4 * dt
        x += dx
        v += dv
        v += torch.randn_like(v) * 8**0.5 * dt**0.5

    # If we are on the last timestep, output the denoised image
    return x - 16 * v * end_step


@torch.no_grad()
def sample_ode(model, x, n_steps, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    # steps = torch.linspace(math.log(1), math.log(1e-3), n_steps).exp()
    # steps = torch.cat([steps, torch.tensor([0.])])
    steps = torch.linspace(1, 0, n_steps + 1)
    x, v = x.chunk(2, dim=1)
    v /= 4**0.5

    # The sampling loop
    for i in trange(len(steps) - 1):

        # Get the model output
        with torch.cuda.amp.autocast():
            eps = model(torch.cat([x, v], dim=1), ts * steps[i], **extra_args).float()

        t = steps[i]
        sigma = make_sigma_hsm(t[None])[0]
        sigma_xx, sigma_xv, sigma_vv = sigma[0, 0], sigma[0, 1], sigma[1, 1]
        l_t = torch.sqrt(sigma_xx / (sigma_xx * sigma_vv - sigma_xv**2))
        score = -eps * l_t  # - v / sigma_vv
        dt = steps[i] - steps[i + 1]
        dx = -16 * v * dt
        dv = (x + score) * 4 * dt
        x += dx
        v += dv

    # If we are on the last timestep, output the denoised image
    return x


from torchdiffeq import odeint


@torch.no_grad()
def sample_ode_rk45(model, x, n_steps, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    # steps = torch.linspace(math.log(1), math.log(1e-3), n_steps).exp()
    # steps = torch.cat([steps, torch.tensor([0.])])
    eps = 1e-2
    time_tensor = torch.tensor([0., 1. - eps], device=x.device)
    x, v = x.chunk(2, dim=1)
    v /= 4**0.5

    u = torch.cat((x, v), dim=1)

    def ode_func(t, u):
        # Get the model output
        with torch.cuda.amp.autocast():
            eps = model(u, ts *
                        (1. - t), **extra_args).float()

        sigma = make_sigma_hsm(1. - t[None])[0]
        sigma_xx, sigma_xv, sigma_vv = sigma[0, 0], sigma[0, 1], sigma[1, 1]
        l_t = torch.sqrt(sigma_xx / (sigma_xx * sigma_vv - sigma_xv**2))
        x, v = u.chunk(2, dim=1)
        score = -eps * l_t  # - v / sigma_vv
        dx_dt = - 16 * v
        dv_dt = 4 * (x + score)
        du_dt = torch.cat((dx_dt, dv_dt), dim=1)

        print(f't: {(1 - t).item():g}, eps: {eps.pow(2).mean().sqrt().item():g}, '
              f'x: {x.pow(2).mean().sqrt().item():g}, v: {v.pow(2).mean().sqrt().item():g} '
              f'dx_dt: {dx_dt.pow(2).mean().sqrt().item():g}, dv_dt: {dv_dt.pow(2).mean().sqrt().item():g} '
              f'l_t: {l_t.item():g}')

        return du_dt

    solution = odeint(ode_func,
                      u,
                      time_tensor,
                      rtol=1e-4,
                      atol=1e-4,
                      method='scipy_solver',
                      options={'solver': 'RK45'})

    u = solution[-1]
    return u.chunk(2, dim=1)[0]


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


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def make_cfg_model_fn(model, cond_scale):
    def cfg_model_fn(x, t, text):
        x_in = x.repeat([2, 1, 1, 1])
        t_in = t.repeat([2])
        null_text_in = model.null_text.repeat([text.shape[0], 1])
        text_in = torch.cat([text, null_text_in])
        v_cond, v_uncond = model(x_in, t_in, text_in).chunk(2, dim=0)
        return v_uncond + (v_cond - v_uncond) * cond_scale
    return cfg_model_fn


def make_sigma_dsm(t, eps=1e-6):
    sigma_xx = torch.exp(16 * t) - 1 - 16 * t - 128 * t**2
    sigma_xv = 64 * t**2
    sigma_vv = torch.expm1(16 * t) / 4 + 4 * t - 2 * (4 * t)**2
    sigma = torch.stack([torch.stack([sigma_xx, sigma_xv], dim=-1), torch.stack([sigma_xv, sigma_vv], dim=-1)], dim=-1)
    return sigma * torch.exp(-16 * t[:, None, None]) + eps * torch.eye(2, device=sigma.device, dtype=sigma.dtype)


def make_sigma_hsm(t, eps=1e-6):
    sigma_vv_0 = 0.01
    sigma_xx = torch.exp(16 * t) - 1 - 16 * t - 128 * t**2 + 16 * (4 * t)**2 * sigma_vv_0
    sigma_xv = 16 * t * sigma_vv_0 + 64 * t**2 - 128 * t**2 * sigma_vv_0
    sigma_vv = torch.expm1(16 * t) / 4 + 4 * t + sigma_vv_0 * (1 + 4 * (4 * t)**2 - 16 * t) - 2 * (4 * t)**2 + eps
    sigma = torch.stack([torch.stack([sigma_xx, sigma_xv], dim=-1), torch.stack([sigma_xv, sigma_vv], dim=-1)], dim=-1)
    l_t = torch.sqrt(sigma_xx / (sigma_xx * sigma_vv - sigma_xv**2))
    return sigma * torch.exp(-16 * t[:, None, None]) + eps * torch.eye(2, device=sigma.device, dtype=sigma.dtype)


def noise_image(x_0, t):
    tb = t[:, None, None, None]
    gamma = 0.04
    # v_0 = torch.randn_like(x_0) * (gamma / 4)**0.5
    v_0 = torch.zeros_like(x_0)
    mean_x = 8 * tb * x_0 + 16 * tb * v_0 + x_0
    mean_v = -4 * tb * x_0 - 8 * tb * v_0 + v_0
    mean = torch.cat([mean_x, mean_v], dim=1) * torch.exp(-8 * tb)
    eps = torch.randn_like(mean)
    sigma = make_sigma_hsm(t)
    scale_tril = torch.linalg.cholesky(sigma.double()).to(x_0.dtype)
    noise_to_add = torch.einsum('ndchw,ned->nechw', torch.stack(eps.chunk(2, dim=1), dim=1), scale_tril)
    sample = noise_to_add.reshape(mean.shape) + mean
    return sample, eps.chunk(2, dim=1)[1]


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    batch_size = 64

    TRAIN_ANN = 'annotations/captions_train2017.json'
    TRAIN_ROOT = 'train2017'
    VAL_ANN = 'annotations/captions_val2017.json'
    VAL_ROOT = 'val2017'

    size_pad = 72
    size = 64

    tf = transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(size_pad, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size_pad),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tok_wrap = TokenizerWrapper()

    def ttf(caption):
        index = torch.randint(len(caption), [])
        return tok_wrap(caption)[index]

    train_set = datasets.CocoCaptions(TRAIN_ROOT, TRAIN_ANN,
                                      transform=tf, target_transform=ttf)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True,
                               num_workers=16, persistent_workers=True, pin_memory=True)

    demo_prompts = Path('coco_demo_prompts.txt').read_text().split('\n')[:64]
    demo_prompts = tok_wrap(demo_prompts).to(device)

    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    model = DiffusionModel().to(device)
    model_ema = deepcopy(model)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    log_file = open('losses.csv', 'w')
    print('epoch', 'i', 'loss', 'loss_var', sep=',', file=log_file, flush=True)

    params = [{'params': model.parameters(), 'lr': 3e-5, 'weight_decay': 1e-4}]
    opt = optim.AdamW(params, eps=1e-5)
    lambda_fn = lambda x: min(1, (x + 1) / 500) * 0.98**(x / len(train_dl))
    scaler = torch.cuda.amp.GradScaler()
    sched = optim.lr_scheduler.LambdaLR(opt, lambda_fn)
    ema_decay = 0.999
    epoch = 0

    def eval_loss(model, rng, images, texts):
        t = rng.draw(images.shape[0])[:, 0].to(device)
        noised_images, eps_targets = noise_image(images, t)

        # Drop out the text on 20% of the examples
        p = torch.rand([texts.shape[0], 1], device=texts.device)
        texts_drop = torch.where(p < 0.2, model.null_text, texts)

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast(False):
            eps = model(noised_images, t, texts_drop)
            losses = (eps - eps_targets).pow(2).flatten(1).mean(1)
            loss = losses.mean()
            loss_var = losses.detach().var(unbiased=False)
            return loss, loss_var

    def train():
        opt.zero_grad()
        for i, (images, texts) in enumerate(tqdm(train_dl)):
            # Evaluate the loss
            loss, loss_var = eval_loss(model, rng, images.to(device), texts.to(device))
            print(epoch, i, loss.item(), loss_var.item(), sep=',', file=log_file, flush=True)

            # Do the optimizer step and EMA update
            # loss.backward()
            scaler.scale(loss).backward()
            # opt.step()

            if i % 1 == 0:
                scaler.unscale_(opt)
                # nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
                ema_update(model, model_ema, 0.99 if epoch < 20 else ema_decay)

            if i % 50 == 0:
                tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}, '
                           f'std: {loss_var.sqrt().item():g}')
            # if i % 500 == 499:
            #     demo()

    @torch.no_grad()
    @eval_mode(model_ema)
    def demo():
        tqdm.write('\nSampling...')
        # x = torch.randn([demo_prompts.shape[0], 6, size, size], device=device)
        x = torch.randn([36, 6, size, size], device=device)
        model_fn = make_cfg_model_fn(model_ema, 3)
        # fakes = sample_ode_rk45(model_fn, x, 250, {'text': demo_prompts[:36]})
        fakes = sample_sde(model_fn, x, 1000, {'text': demo_prompts[:36]})
        grid = utils.make_grid(fakes, 6, padding=0)
        TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(f'demo_{epoch:05}.png')
        tqdm.write('')

    def save():
        filename = 'mscoco_3_cld_test.pth'
        obj = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'sched': sched.state_dict(),
            'epoch': epoch,
        }
        torch.save(obj, filename)

    try:
        # demo()
        while True:
            print('Epoch', epoch)
            train()
            epoch += 1
            if epoch % 1 == 0:
                demo()
            save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
