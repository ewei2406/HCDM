import torch

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
from tqdm import tqdm

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        imgs.append(img.cpu().numpy())
    return imgs

import torch.nn.functional as F

@torch.no_grad()
def sample(model, image_size, scheduler, timesteps, batch_size=16, channels=3):

    # define alphas 
    betas = scheduler(timesteps=timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return p_sample_loop(
        model, 
        (batch_size, channels, image_size, image_size), 
        timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)