import torch
import numpy as np
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH  = WIDTH  // 8
LATENT_HEIGHT = HEIGHT // 8

@torch.no_grad()
def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True, # classifier free guidance
    cfg_scale=7.5,
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    assert (strength > 0) and (strength <= 1)
    if idle_device:
        to_idle = lambda x: x.to(idle_device)
    else:
        to_idle = lambda x: x.to("cpu")

    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    clip = models['clip']
    clip.to(device)

    # TEXT CONTEXT

    if do_cfg:
        # convert to list
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding='max_length', max_length=77
        ).input_ids
        # [B, N, D]
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)

        # convert to list
        uncond_tokens = tokenizer.batch_encode_plus(
            [uncond_prompt], padding='max_length', max_length=77,
        ).input_ids
        # [B, N, D]
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)

        # combine into one batch [2B, N, D]
        context = torch.cat([cond_context, uncond_context], dim=0)

    else:
        # convert to list
        tokens = tokenizer.batch_encoder_plus(
            [prompt], padding='max_length', max_length=77,
        ).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        # [B, N, D]
        context = clip(tokens)

    to_idle(clip) # offload back to CPU

    # sampler
    sampler = DDPMSampler(generator)
    sampler.set_inference_timesteps(n_inference_steps)

    LATENT_SHAPE = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

    if input_image:
        encoder = models["encoder"]
        encoder.to(device)

        # [H, W, C]
        image = np.array(input_image.resize((WIDTH, HEIGHT)))
        image = torch.tensor(image, dtype=torch.float, device=device)
        image = rescale(image, (0, 255), (-1, 1))
        image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, C, H, W]

        noise = torch.randn(LATENT_SHAPE, generator=generator, device=device)
        latent = encoder(image, noise) # [1, 4, H/8, W/8]

        sampler.set_strength(strength=strenth)
        latent = sampler.add_noise(latent, sampler.timesteps[0])

        to_idle(encoder)
    else:
        latent = torch.randn(LATENT_SHAPE, generator=genreator, device=device)

    # diffusion model
    diffusion = models["diffusion"]
    diffusion.to(device)


    return
#
