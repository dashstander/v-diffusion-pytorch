#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import partial
import math
from pathlib import Path

from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm import trange
from tqdm.std import tqdm
import wandb


p = argparse.ArgumentParser()
p.add_argument(
    '--train-set',
    type=Path,
    required=True,
    help='the training set location')
p.add_argument('--batch-size', type=int, default=2)
p.add_argument('--run-name', type=str)
p.add_argument('--seed', type=int, default=21)
p.add_argument('--grad-accum', type=int, default=8)


# Define utility functions

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

    def forward(self, input):
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
        cs = [c, c , c * 2, c * 4, c * 4, c * 4, c * 8]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(128, 1024, 1024),
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
            # conv_block(cs[0], cs[0], cs[0]),
            # conv_block(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 128x128
                conv_block(cs[0], cs[1], cs[1]),
                # conv_block(cs[1], cs[1], cs[1]),
                # conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 64x64
                    conv_block(cs[1], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    # conv_block(cs[2], cs[2], cs[2]),
                    # conv_block(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,  # 32x32
                        conv_block(cs[2], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        # conv_block(cs[3], cs[3], cs[3]),
                        # conv_block(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,  # 16x16
                            conv_block(cs[3], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            # conv_block(cs[4], cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            # conv_block(cs[4], cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            SkipBlock([
                                self.down,  # 8x8
                                conv_block(cs[4], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                #SelfAttention2d(cs[5], cs[5] // 64),
                                #conv_block(cs[5], cs[5], cs[5]),
                                #SelfAttention2d(cs[5], cs[5] // 64),
                                #conv_block(cs[5], cs[5], cs[5]),
                                #SelfAttention2d(cs[5], cs[5] // 64),
                                # SkipBlock([
                                #    self.down,  # 4x4
                                #    conv_block(cs[5], cs[6], cs[6]),
                                #    SelfAttention2d(cs[6], cs[6] // 64),
                                #    conv_block(cs[6], cs[6], cs[6]),
                                #    SelfAttention2d(cs[6], cs[6] // 64),
                                    # conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    # conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    # conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    # conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                    # conv_block(cs[6], cs[6], cs[6]),
                                    # SelfAttention2d(cs[6], cs[6] // 64),
                                #    conv_block(cs[6], cs[6], cs[5]),
                                #    SelfAttention2d(cs[5], cs[5] // 64),
                                #    self.up,
                                #]),
                                #conv_block(cs[5] * 2, cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[5]),
                                SelfAttention2d(cs[5], cs[5] // 64),
                                # conv_block(cs[5], cs[5], cs[5]),
                                # SelfAttention2d(cs[5], cs[5] // 64),
                                conv_block(cs[5], cs[5], cs[4]),
                                SelfAttention2d(cs[4], cs[4] // 64),
                                self.up,
                            ]),
                            conv_block(cs[4] * 2, cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[4]),
                            SelfAttention2d(cs[4], cs[4] // 64),
                            # conv_block(cs[4], cs[4], cs[4]),
                            # SelfAttention2d(cs[4], cs[4] // 64),
                            conv_block(cs[4], cs[4], cs[3]),
                            SelfAttention2d(cs[3], cs[3] // 64),
                            self.up,
                        ]),
                        conv_block(cs[3] * 2, cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[3]),
                        # conv_block(cs[3], cs[3], cs[3]),
                        conv_block(cs[3], cs[3], cs[2]),
                        self.up,
                    ]),
                    conv_block(cs[2] * 2, cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[2]),
                    # conv_block(cs[2], cs[2], cs[2]),
                    conv_block(cs[2], cs[2], cs[1]),
                    self.up,
                ]),
                conv_block(cs[1] * 2, cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[1]),
                # conv_block(cs[1], cs[1], cs[1]),
                conv_block(cs[1], cs[1], cs[0]),
                self.up,
            ]),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            # conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 3, is_last=True),
        )

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5**0.5

    def forward(self, input, t):
        mapping_timestep_embed = self.mapping_timestep_embed(t[:, None])
        self.state['cond'] = self.mapping(mapping_timestep_embed)
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


class ImageDataset(data.Dataset):
    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""
    
    def __init__(self, folder, preprocess_im, preprocess_text=None, enable_text=True, enable_image=True):
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
            else:
                text = None
        except (Image.UnidentifiedImageError, OSError, KeyError, Image.DecompressionBombError,):
            print(f"Failed to load image/text {key}. Skipping.")
            return None  # return None to be filtered in the batch collate_fn
        return image_tensor



class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def eval_step(model, batch, ):
    
    images = batch.to('cuda')
    # Sample timesteps
    t = torch.rand(images.shape[0]).to('cuda')
    # Calculate the noise schedule parameters for those timesteps
    alphas, sigmas = get_alphas_sigmas(t)
    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]
    noise = torch.randn_like(images)
    noised_images = images * alphas + noise * sigmas
    targets = noise * alphas - images * sigmas
    # Compute the model output and the loss.
    # with torch.cuda.amp.autocast():
    pred = model(noised_images, t)
    loss = F.mse_loss(pred, targets)
    return loss


def train_step(model, batch):
    loss = eval_step(model, batch)
    log_dict = {'train/loss': loss}
    wandb.log(log_dict)
    return loss


def get_models():
    diffusion_model = DiffusionModel()
    model_ema = deepcopy(diffusion_model)
    return diffusion_model, model_ema

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
def on_batch_end(model, global_step):
   

    noise = torch.randn([16, 3, 256, 256], device=model.device)
    model.eval()
    fakes = sample(model, noise, 1000, 1)

    grid = utils.make_grid(fakes, 4, padding=0).cpu()
    image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
    filename = f'demo_{global_step:08}.png'
    image.save(filename)
    log_dict = {
        'demo_grid': wandb.Image(image)
    }
    wandb.log(log_dict)
    model.train()



def get_dataloader(train_fp, size, batch_size):
    
    tf = transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = ImageDataset(train_fp, tf, enable_text=False)
    train_dl = data.DataLoader(
        train_set,
        batch_size,
        sampler=data.RandomSampler(train_set),
        num_workers=6,
        persistent_workers=True
    )
    return train_dl


def train_loop(model, ema, optimizer, data_loader, accum_iter, init_step=0):
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(data_loader)):
        loss = train_step(model, batch)
        loss /= accum_iter
        loss.backward()
        #optimizer.step()
        if ((i + 1) % accum_iter == 0) or ((i + 1) == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
        do_ema_upodate(model, ema, i + init_step)
        if (i + init_step) % 10000 == 0 and i > 0:
            torch.save(
                {
                    'model': model,
                    'ema': ema,
                    'optimizer': optimizer,
                    'rng_state': torch.random.get_rng_state() 
                },
                f'./checkpoints/diffusion_cc3m/ckpt_{i+init_step}.pt'
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

    model, ema_model = get_models()
    num_params = param_count(model)
    print(f'Diffusion model # of parameters: {num_params}')
    wandb.config.update({'num_params': num_params})
    opt = get_optimizer(model)
    wandb.watch(model)
    global_counter = 0
    model.to('cuda')
    ema_model.to('cuda')
    for _ in range(100):
        global_counter += train_loop(
            model,
            ema_model,
            opt,
            data,
            args.grad_accum,
            global_counter
        )
        


if __name__ == '__main__':
    main()
