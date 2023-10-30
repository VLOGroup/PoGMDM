import imageio
import numpy as np
import torch as th
import torchvision.utils as tvu

import data
import models

sigmas = [0, 0.025, 0.05, 0.1]

kernel_size = 7
n_f = kernel_size**2 - 1
bs = 64 * 4000
patch_size = kernel_size
color = False
rotate = True
flip = True
n_w = 63 * 2 - 1

sigmas = [0, .025, .05, .1, .2]
dataset = data.BSDS(color, bs, patch_size, rotate, flip)
R = models.ProductGMM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    symmetric=True,
    ortho=True,
    vmin=-1,
    vmax=1,
    kernel_size=kernel_size,
    K_init='random',
    n_w=n_w,
    w_init='student-t',
    sigmas=th.Tensor(sigmas)
).cuda()
th.set_grad_enabled(False)

state = R.load_state_dict(th.load('./out/patch/state_final.pth'))

for i_sigma, sigma in enumerate(sigmas):
    for y in dataset:
        break
    # R.set_sigma(sigma)
    # y = th.load(f'generated_{sigma:.3f}.pth').cuda()
    # R.grad(y + sigma * th.randn_like(y))

    y = y[i_sigma * 50:(i_sigma + 1) * 50] + sigma * th.randn_like(y)[:50]
    imageio.imsave(
        f'./out/patch/sampling/true_{sigma:.3f}.png',
        (np.clip(
            tvu.make_grid(
                y - y.mean((1, 2, 3), keepdim=True) + 0.5,
                nrow=10,
                padding=1,
                pad_value=1,
            ).permute(1, 2, 0).cpu().numpy(), 0, 1
        ) * 255.).astype(np.uint8)
    )

for sigma in sigmas:
    generated = th.from_numpy(
        np.load(f'./out/patch/sampling/generated_analytical_{sigma:.3f}.npy')
    )[:, None]
    generated = generated[:50]
    imageio.imsave(
        f'./out/patch/sampling/analytical_{sigma:.3f}.png',
        (tvu.make_grid(
            generated - generated.mean((1, 2, 3), keepdim=True) + 0.5,
            nrow=10,
            padding=1,
            pad_value=1
        ).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.).astype(np.uint8)
    )
