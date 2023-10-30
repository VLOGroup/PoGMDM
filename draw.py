import matplotlib.pyplot as plt
import numpy as np
import torch as th

import data
import models

plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams['axes.titlepad'] = 2

sigmas = [0, .025, .05, .1, .2]
N = len(sigmas)
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, N))
)

kernel_size = 7
n_f = kernel_size**2 - 1
bs = 64 * 4000
patch_size = kernel_size
color = False
rotate = True
flip = True
n_w = 63 * 2 - 1
n_scales = 20

dataset = data.BSDS(color, bs, patch_size, rotate, flip)
gamma = 1.
R_gsm = models.ProductGSM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    ortho=True,
    n_scales=n_scales,
    kernel_size=kernel_size,
    K_init='random',
).cuda()
th.set_grad_enabled(False)

state = th.load('./out/gsm/state_final.pth')
state['w.w'] = state['w']
R_gsm.load_state_dict(state)
R_gmm = models.ProductGMM(
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

state = th.load('./out/patch/state_final.pth')
R_gmm.load_state_dict(state)
ylims_f_ = {
    'gmm': [-.2, 10],
    'gsm': [-.2, 6],
}
ylims_fp_ = {
    'gmm': [-20, 20],
    'gsm': [-9.5, 9.5],
}
ylims_tweedie_ = {
    'gmm': [-1, 1],
    'gsm': [-1, 1],
}
for y in dataset:
    break
dm = 0.01
bin_edges = th.linspace(
    -gamma - dm / 2,
    gamma + dm / 2,
    n_w + 1,
).cuda()
x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
for R, name in zip([R_gmm, R_gsm], ['gmm', 'gsm']):
    hist = th.zeros((len(sigmas), n_f, n_w)).cuda()
    ylims_f = ylims_f_[name]
    ylims_fp = ylims_fp_[name]
    ylims_tweedie = ylims_tweedie_[name]
    for i_s, sigma in enumerate(sigmas):
        R.set_sigma(sigma)
        Kx = R.K(y + sigma * th.randn_like(y))

        for k in range(n_f):
            hist[
                i_s,
                k] = th.histogram(Kx[:, k].reshape(-1).cpu(),
                                  bin_edges.cpu())[0].to('cuda')

    fs = []
    fps = []
    K = R.K.weight.data
    scale = 1.1

    n_points = 20
    x = th.linspace(
        -scale * gamma,
        scale * gamma,
        n_points**2,
        dtype=K.dtype,
        device=K.device,
    )[None].repeat(n_f, 1)
    for sig in sigmas:
        R.set_sigma(sig)
        f, fp = R.pot_act(x.view(1, n_f, n_points, n_points))
        f = f.view(n_f, n_points * n_points)
        fp = fp.view(n_f, n_points * n_points)
        fs.append(f)
        fps.append(fp)

    x = x[0]

    fs = th.stack(fs).permute(1, 0, 2)
    fps = th.stack(fps).permute(1, 0, 2)
    norm_k = (K**2).sum((1, 2, 3))
    indices = th.sort(norm_k)[1]
    K = K[indices]
    fs = fs[indices]
    fps = fps[indices]
    hist = hist.permute(1, 0, 2)
    hist = hist[indices]

    fig_k, ax_k = plt.subplots(3, 16, figsize=(16, 3))
    fig_f, ax_f = plt.subplots(3, 16, figsize=(16, 3))
    fig_fp, ax_fp = plt.subplots(3, 16, figsize=(16, 3))
    fig_tweedie, ax_tweedie = plt.subplots(3, 16, figsize=(16, 3))
    fig_h, ax_h = plt.subplots(3, 16, figsize=(16, 3))

    for i, (ff, ffp, hh, kk) in enumerate(zip(fs, fps, hist, K)):
        r, c = divmod(i, 16)
        for sigma, fff, fffp, hhh in zip(sigmas, ff, ffp, hh):
            neg_log = -th.log(hhh).detach().cpu().numpy()
            neg_log -= neg_log.min()
            fff -= fff.min()
            ax_h[r, c].plot(x_hist.cpu(), neg_log)
            ax_f[r, c].plot(x.cpu(), fff.cpu())
            ax_fp[r, c].plot(x.cpu(), fffp.cpu())
            ax_tweedie[r, c].plot(
                x.cpu(),
                x.cpu().numpy() - sigma**2 * fffp.cpu().numpy()
            )
            for axx, ylims in zip([ax_f, ax_h, ax_fp, ax_tweedie],
                                  [ylims_f, ylims_f, ylims_fp, ylims_tweedie]):
                axx[r, c].set_ylim(ylims)
                axx[r, c].grid(True)
                if (r, c) == (2, 0):
                    xt = axx[r, c].get_xticklabels()
                if (r, c) != (2, 0):
                    axx[r, c].tick_params(tick1On=False)
                    axx[r, c].set_xticklabels([])
                    axx[r, c].set_yticklabels([])
                    axx[r, c].set_frame_on(False)

        k_plot = ax_k[r, c].imshow(kk.cpu().squeeze(), cmap='gray')
        ax_k[r, c].axis('off')
        ax_k[r, c].set_title(
            f'\\( [{kk.min().item()*10:.1f}, {kk.max().item()*10:.1f}] \\)',
            fontsize=8
        )

    fig_f.savefig(f'./out/{name}/plots/f.pdf')
    fig_fp.savefig(f'./out/{name}/plots/fp.pdf')
    fig_k.savefig(f'./out/{name}/plots/k.pdf')
    fig_h.savefig(f'./out/{name}/plots/h.pdf')
plt.show()
