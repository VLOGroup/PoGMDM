import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as th

import data
import util

colormap = plt.get_cmap('inferno')

th.set_grad_enabled(False)
color = False
images = data.Set68(color=False).data()[:15]
M, N = images.shape[2:]

# Ordering: GMM7, GMM15, GSM7, wavelet db2, wavelet db4, shearlet
rs = util.get_models(320)

kernel_size = 7
div7 = util.patch2image(
    util.image2patch(th.ones((M, N)).cuda(), (kernel_size, kernel_size)),
    (M, N), (kernel_size, kernel_size)
)
kernel_size = 15
div15 = util.patch2image(
    util.image2patch(th.ones((M, N)).cuda(), (kernel_size, kernel_size)),
    (M, N), (kernel_size, kernel_size)
)
ssim = util.SSIM().cuda()
diff_mult = 3

# Parameters for stochastic denoising
sigma_L = 0.01
L = 250
T = 3


def stochastic_image_denoiser(y, score, sigma):
    '''https://arxiv.org/pdf/2101.09552.pdf'''
    x = y.clone()
    for i_s, sigma in enumerate(sigmas):
        alpha = epsilon * sigma**2 / sigmas[-1]**2
        for t in range(T):
            z = th.randn_like(y)
            delta = score(x, sigma) + (y - x) / (sigma_0**2 - sigma**2)
            x = x + alpha * delta + np.sqrt(2 * alpha) * z

    return x


def tweedie(y, score, sigma):
    return y + score(y, sigma) * sigma**2


for restore_name, restore_fn in zip(
    ['tweedie', 'stoch'],
    [tweedie, stochastic_image_denoiser],
):
    for eval_method in [util.psnr, ssim]:
        for i_s, sigma in enumerate([0.025, 0.05, 0.1, 0.2]):
            epsilon = 5e-6
            sigma_0 = sigma
            gamma = (sigma_L / sigma_0)**(1 / L)
            sigmas = [sigma_0 * gamma**ll for ll in range(1, L + 1)]
            noisy = images + sigma * th.randn_like(images)
            print(f'{sigma:.3f}', end=' & ')
            print(f'{eval_method(images, noisy):.2f}', end=' & ')
            for i_n, (n, gt) in enumerate(zip(noisy, images)):
                diff = (th.abs(n - gt) * diff_mult).cpu().numpy()
                diff_heat = colormap(diff)
                imageio.imsave(
                    f'./out/denoising/{sigma:.3f}/{restore_name}/noisy/{i_n:03d}_d.png',
                    (diff_heat.clip(0, 1).squeeze() * 255.).astype(np.uint8)
                )
                imageio.imsave(
                    f'./out/denoising/{sigma:.3f}/{restore_name}/noisy/{i_n:03d}.png',
                    (n.cpu().numpy().clip(0, 1).squeeze() *
                     255.).astype(np.uint8)
                )
            for i_n, gt in enumerate(images):
                imageio.imsave(
                    f'./out/denoising/{sigma:.3f}/{restore_name}/gt/{i_n:03d}.png',
                    (gt.cpu().numpy().clip(0, 1).squeeze() *
                     255.).astype(np.uint8)
                )
            for i_r, (R, div, name) in enumerate(
                zip(
                    rs, [div7] + [div15] + [div7] + 3 * [th.ones_like(div7)], [
                        'gmm7', 'gmm15', 'gsm7', 'wavelet-db2', 'wavelet-db4',
                        'shearlet'
                    ]
                )
            ):
                R.set_sigma(sigma)

                def score(x, sigma):
                    R.set_sigma(sigma)
                    return -R.grad(x)[1] / div

                denoised = restore_fn(noisy, score, sigma)
                end = ' & ' if i_r < 5 else ' \\\\'
                print(f'{eval_method(images, denoised):.2f}', end=end)

                for i_n, den in enumerate(denoised):
                    imageio.imsave(
                        f'./out/denoising/{sigma:.3f}/{restore_name}/{name}/{i_n:03d}.png',
                        (den.cpu().numpy().clip(0, 1).squeeze() *
                         255.).astype(np.uint8)
                    )
                for i_n, (den, gt) in enumerate(zip(denoised, images)):
                    diff = (th.abs(den - gt) * diff_mult).cpu().numpy()
                    diff_heat = colormap(diff)
                    imageio.imsave(
                        f'./out/denoising/{sigma:.3f}/{restore_name}/{name}/{i_n:03d}_d.png',
                        (diff_heat.clip(0, 1).squeeze() *
                         255.).astype(np.uint8)
                    )
            print()
        print('\\midrule')
    print('{\\#Params} & & ', end='')
    for i_r, R in enumerate(rs):
        end = ' & ' if i_r < 5 else ''
        print(f'\\num{{{sum(p.numel() for p in R.parameters())}}}', end=end)
    print('\\\\\\bottomrule')
