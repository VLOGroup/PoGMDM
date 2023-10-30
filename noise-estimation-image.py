import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as th

import data
import models
import util

plt.rcParams.update({
    "text.usetex": True,
})

n_f = 48
bs = 1
patch_size = 7
kernel_size = 7
color = False
rotate = True
flip = True
log_freq = 50
n_w = 63 * 2 - 1
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
).cuda()
R.load_state_dict(th.load('./out/patch/state_final.pth'))
th.set_grad_enabled(False)

for i in range(10):
    image = data.Set68().data()[i:i + 1].squeeze()

    sigma_1 = .2
    sigma_2 = .1
    tile_size = 35
    mask = th.from_numpy(
        np.tile(
            np.vstack((
                np.hstack((
                    np.ones((tile_size, tile_size)),
                    np.zeros((tile_size, tile_size))
                )),
                np.hstack((
                    np.zeros((tile_size, tile_size)),
                    np.ones((tile_size, tile_size))
                ))
            )), (20, 20)
        )
    )[:image.shape[0], :image.shape[1]]
    mask = (
        1 - th.from_numpy(
            np.tile(mask, (10, 10))[:image.shape[0], :image.shape[1]]
        ).cuda()
    )

    noise = th.randn_like(image)
    noisy = image + (sigma_2 * mask + sigma_1 * (1 - mask)) * noise

    patches = util.image2patch(noisy.squeeze(),
                               (kernel_size, kernel_size
                                )).view(-1, 1, kernel_size,
                                        kernel_size).cuda().float()

    sigmas_est = np.linspace(0, .5, 200)
    energy = np.empty((patches.shape[0], sigmas_est.shape[0]))

    for i_e, sigma_est in enumerate(sigmas_est):
        R.set_sigma(sigma_est)
        energy[:, i_e] = R.grad(patches)[0].cpu().numpy().squeeze()

    estimates_flat = sigmas_est[energy.argmin(1)]
    estimates = sigmas_est[energy.argmin(1)].reshape(
        (image.shape[0] - 6, image.shape[1] - 6)
    )
    print(estimates_flat.shape)
    print(estimates.min(), estimates.max())
    print(mask.shape, mask.dtype)
    imageio.imwrite(
        f'./out/blind/{i}/mask.png',
        mask.cpu().numpy().astype(np.uint8)
    )
    imageio.imwrite(
        f'./out/blind/{i}/noisy.png',
        (noisy.cpu().clip(0, 1).numpy() * 255).astype(np.uint8)
    )
    imageio.imwrite(
        f'./out/blind/{i}/estimate.png',
        (estimates / estimates.max() * 255.).astype(np.uint8)
    )

    denoised_patches = []
    for i_p, p in enumerate(patches):
        print(i_p)
        R.set_sigma(estimates_flat[i_p].item())
        denoised_patches.append(
            p - R.grad(p[None])[1] * estimates_flat[i_p]**2
        )

    print(util.psnr(noisy[None, None], image[None, None]))

    M, N = image.shape
    div = util.patch2image(
        util.image2patch(th.ones((M, N)).cuda(), (kernel_size, kernel_size)),
        (M, N), (kernel_size, kernel_size)
    )
    denoised_patches = th.stack(denoised_patches).squeeze()
    blind_denoised = util.patch2image(
        denoised_patches.view(-1, 7 * 7), image.shape, (7, 7)
    ) / div
    print(util.psnr(blind_denoised[None, None], image[None, None]))
    imageio.imwrite(
        f'./out/blind/{i}/denoised.png',
        (blind_denoised.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    )
    cm = plt.get_cmap('inferno')
    diff = ((blind_denoised - image).abs() * 3).cpu().numpy().clip(0, 1)
    diff_cm = cm(diff)
    imageio.imwrite(
        f'./out/blind/{i}/diff.png', (diff_cm * 255).astype(np.uint8)
    )
    print(denoised_patches.shape)
