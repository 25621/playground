"""Diffusers quickstart: unconditional image generation.

Generates a 256x256 image of a cat with google/ddpm-cat-256.
The default DDPM sampler needs 1000 steps; we swap in DDIM so
50 steps are enough, which keeps CPU inference fast.

Run:
    .venv/bin/python quickstart.py
"""

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline

pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
pipe = DDIMPipeline(unet=pipe.unet, scheduler=DDIMScheduler.from_config(pipe.scheduler.config))

generator = torch.Generator().manual_seed(42)
image = pipe(num_inference_steps=50, generator=generator).images[0]

image.save("cat.png")
print(f"Saved {image.size[0]}x{image.size[1]} image to cat.png")
