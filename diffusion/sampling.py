import torch
from tqdm.auto import trange

from . import utils


@torch.no_grad()
def sample(model, x, steps, eta, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
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


@torch.no_grad()
def cond_sample(model, x, steps, eta, extra_args, cond_fn, callback=None):
    """Draws guided samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                v = model(x, ts * steps[i], **extra_args)

            pred = x * alphas[i] - v * sigmas[i]

            # Call the callback
            if callback is not None:
                callback({'x': x, 'i': i, 't': steps[i], 'v': v.detach(), 'pred': pred.detach()})

            if steps[i] < 1:
                cond_grad = cond_fn(x, ts * steps[i], pred, **extra_args).detach()
                v = v.detach() - cond_grad * (sigmas[i] / alphas[i])
            else:
                v = v.detach()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
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


@torch.no_grad()
def reverse_sample(model, x, steps, extra_args, callback=None):
    """Finds a starting latent that would produce the given image with DDIM
    (eta=0) sampling."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps) - 1, disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    return x


def phi_transfer_step(x, eps, alpha, alpha_next):
    # N.B. not clear if this is alpha _the noise_ or alpha _the signal_ --> just doing the formula
    x_part = torch.sqrt(alpha_next / alpha) * x
    eps_part_num = alpha_next - alpha
    eps_part_denom = (
        torch.sqrt(alpha) * 
        (
            torch.sqrt((1 - alpha_next) * alpha) + 
            torch.sqrt((1 - alpha) * alpha_next)
        )
    )
    eps_part = (eps_part_num / eps_part_denom) * eps
    return x_part - eps_part



def model_one_step(model, x, step, alpha, sigma, extra_args):
    ts = x.new_ones([x.shape[0]])
    # Get the model output (v, the predicted velocity)
    with torch.cuda.amp.autocast():
        v = model(x, ts * step, **extra_args).float()
    # Predict the noise and the denoised image
    pred = x * alpha - v * sigma
    eps  = x * sigma + v * alpha
    return pred, eps


def linear_multistep_update(x, eps, prev_eps, alpha, alpha_next):
    eps_prev0 = prev_eps[0]
    eps_prev1 = prev_eps[1]
    eps_prev2 = prev_eps[2]
    eps_lms = (55 * eps - 59 * eps_prev0 + 37 * eps_prev1 - 9 * eps_prev2)
    pred = phi_transfer_step(x, eps_lms, alpha, alpha_next)
    return pred


def rk_update(model, x, eps, steps, index, extra_args):
    t, t_next = steps[index], steps[index + 1]
    t_mid = (t_next + t) / 2
    alphas, sigmas = utils.t_to_alpha_sigma(torch.linspace(t, t_next, 3)) # calculate alpha & sigma for current time, next time, and midpoint
    sigma, sigma_mid, sigma_next = sigmas[0], sigmas[1], sigmas[2]
    alpha, alpha_mid, alpha_next = alphas[0], alphas[1], alphas[2]
    eps_1 = eps
    x_1 = phi_transfer_step(x, eps_1, alpha, alpha_mid)
    _, eps_2 = model_one_step(model, x_1, t_mid, alpha_mid, sigma_mid, extra_args)
    x_2 = phi_transfer_step(x, eps_2, alpha, alpha_mid)
    _, eps_3 = model_one_step(model, x_2,  t_mid, alpha_mid, sigma_mid, extra_args)
    x_3 = phi_transfer_step(x, eps_3, alpha, alpha_mid)
    _, eps_4 = model_one_step(model, x_3, t_mid, alpha_mid, sigma_mid, extra_args)
    eps_rk = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    pred = phi_transfer_step(x, eps_rk, alpha, alpha_next)
    return pred


@torch.no_grad()
def pdsn_sample(model, x, steps, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    eps_cache = []

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:

            if len(eps_cache) < 3:
                pred = rk_update(model, pred, eps, steps, i, extra_args)
            else:
                pred = linear_multistep_update(pred, eps, eps_cache, alphas[i], alphas[i+1])
                eps_cache.pop(0)
            eps_cache.append(eps)

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred # is x just equal to pred? * alphas[i + 1] + eps * sigmas[i + 1]

    # If we are on the last timestep, output the denoised image
    return pred