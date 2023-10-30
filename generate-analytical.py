import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.utils as tvu

import models

rng = np.random.default_rng()

kernel_size = 7
n_f = kernel_size**2 - 1
bs = 64 * 4000
patch_size = kernel_size
color = False
rotate = True
flip = True
n_w = 63 * 2 - 1

sigmas = [0, .025, .05, .1]
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

K = R.K.weight.cpu().numpy()
weights = R.w.get().cpu().numpy()
sigma_0 = R._sigma_0
mus = R.mus.cpu().numpy()
n_patch = 1_000_000
patch = np.zeros((n_patch, 7, 7))

for sigma in [0., 0.025, 0.05, 0.1, 0.2]:
    R.set_sigma(sigma)
    patch = np.zeros((n_patch, 7, 7))
    for i_k, k in enumerate(K):
        w_j = weights[i_k]
        i = np.random.choice(w_j.shape[0], size=(n_patch, ), p=w_j)
        var = (sigma_0**2 + sigma**2 * (k**2).sum())
        sample = rng.normal(size=(n_patch, )) * np.sqrt(var) + mus[i]
        patch += sample[:, None, None] * k[0] / (k**2).sum()

    np.save(f'./out/patch/sampling/generated_analytical_{sigma:.3f}.npy', patch)
