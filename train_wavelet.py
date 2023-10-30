import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_wavelets as pwt
import torch as th
import torchvision.utils as tvu
import logsumexp
from torch.utils.tensorboard.writer import SummaryWriter

import data
import models
import optim
import util

bs = 250
log_freq = 10
n_w = 125
wave = 'db2'
levels = 2
flip = True
rotate = True
color = False
patch_size = 64
K = pwt.DWTForward(wave=wave, J=levels, mode='reflect').cuda().to(th.float64)
mus_each = []
xs_pot = []
dataset = data.BSDS(color, bs, patch_size, rotate, flip)
for y in dataset:
    break
n_points = 20
for level in range(levels):
    for direction in range(3):
        qs = th.quantile(
            K(y.to(th.float64))[1][level][:, :, direction], 0.999
        ) * 1.1
        qs = 1
        mus_each.append(th.linspace(-qs, qs, n_w, device='cuda')[None])
        xs_pot.append(
            th.linspace(-qs, qs, n_points**2, device='cuda', dtype=th.float64)
        )
mus = th.cat(mus_each, dim=0)
gamma = .5

R = models.WaveletGMM(
    mus=mus,
    levels=levels,
    vmin=-gamma,
    vmax=gamma,
    n_w=n_w,
    w_init='student-t',
    im_sz=64,
    wave=wave,
).cuda().to(th.float64)
print(R.w.get().shape)

R.set_sigma(0)
scale = 1.
ipalm = False
# R.load_state_dict(th.load(f'./out/wavelets/db2/state_009000.pth'))

t = str(time.time())
writer = SummaryWriter(log_dir=f'./log/wavelets/{wave}/' + t)
lrs = {
    'w.w': 1e-5,
    'h': 1e-4,
    'lambdas': 5e-4,
    # 'global_gmm.precs_chol_flat': 5e-3,
    # 'global_gmm.mus': 5e-2,
    # 'global_gmm.w': 5e-2,
}
if ipalm:
    groups = []
    for k, v in R.named_parameters():
        if k not in lrs.keys():
            continue
        groups.append({
            'params': v,
            'name': k,
        })
    optimizer = optim.IPalm(groups)
else:
    groups = []
    for k, v in R.named_parameters():
        print(k)
        groups.append({
            'params': v,
            'lr': lrs[k],
            'name': k,
        })
    optimizer = optim.AdaBelief(groups)


def loss_criterion(y, R, sigmas):
    n = [th.randn_like(y) for _ in range(len(sigmas))]

    def closure(compute_grad=False):
        with th.set_grad_enabled(compute_grad):
            loss_sm = 0
            for sigma, noise in zip(sigmas, n):
                R.set_sigma(sigma)
                x = y + sigma * noise
                loss_sm += ((sigma * R.grad(x)[1] - noise)**2).sum()
            return [loss_sm / bs / len(sigmas)]

    return closure


i = 0
for i, y in enumerate(dataset):
    y = y.to(th.float64)
    closure = loss_criterion(y, R, 0.025 + np.random.rand(5) * 0.2)

    writer.add_scalar('loss/score', sum(closure()).item(), global_step=i)
    with th.no_grad():
        if i % log_freq == 0:
            # global_means = R.global_gmm.mus.data.clone()
            # sz = int(np.sqrt(global_means.shape[1]))
            # global_means = global_means.view(global_means.shape[0], 1, sz, sz)
            # global_means = tvu.make_grid(global_means, nrow=8)
            # writer.add_image(
            #     'means',
            #     tvu.make_grid(
            #         global_means, nrow=8, normalize=True, scale_each=True
            #     ),
            #     global_step=i
            # )
            R.set_sigma(0)
            fig, ax = plt.subplots(levels, 3)
            # fig_h, ax_h = plt.subplots(2, 3)
            dm = 0.01
            for level in range(levels):
                for direction in range(3):
                    weights = R.w.get()[level * 3 + direction:level * 3 +
                                        direction + 1]
                    for sigma in [0, .025, .05, .1, .2]:
                        R.set_sigma(sigma)

                        #     Kx = R.K(y + sigma * th.randn_like(y))
                        #     bin_edges = th.linspace(
                        #         R.mus[level * 3 + direction, 0] - dm / 2,
                        #         R.mus[level * 3 + direction, -1] + dm / 2,
                        #         n_w + 1,
                        #     ).cuda()
                        #     x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
                        #     hist = th.histogram(
                        #             Kx[1][level][:, :, direction].reshape(-1).cpu(), bin_edges.cpu()
                        #     )[0].to('cuda')
                        #     nlogh = -th.log(hist).cpu().numpy()
                        #     nlogh -= nlogh.min()
                        #     ax_h[level, direction].plot(x_hist.cpu(), nlogh)
                        pot, _ = logsumexp.pot_act(
                            R.lambdas[level * 3 + direction] *
                            xs_pot[level * 3 +
                                   direction].view(1, 1, n_points, n_points),
                            weights, R.mus[level * 3 + direction].clone(),
                            R.sigma[level * 3 + direction][None]
                        )
                        pot = pot.view(n_points * n_points).detach()
                        pot -= pot.min()
                        pot = pot.cpu().numpy()
                        pot[pot > 7] = np.nan
                        ax[level, direction].plot(
                            xs_pot[level * 3 + direction].cpu().numpy(), pot
                        )

            writer.add_figure('pot', fig, global_step=i)
            # writer.add_figure('hist', fig_h, global_step=i)

            for s in [.025, .05, .1, .2]:
                R.set_sigma(s)
                x = y[:25] + s * th.randn_like(y[:25])
                y_hat = x - R.grad(x)[1] * s**2
                stack = th.concat((x[:8], y_hat[:8], y[:8]), dim=0)
                writer.add_scalar(
                    f'psnr {s:.3f}', util.psnr(y[:25], y_hat), global_step=i
                )
                writer.add_image(
                    f'test {s:.3f}',
                    tvu.make_grid(th.clip(stack, 0, 1), nrow=8),
                    global_step=i
                )
                R.set_sigma(0)

            th.save(R.state_dict(), f'./out/wavelets/{wave}/state_{i:06d}.pth')

    if ipalm:
        loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        loss = closure(True)
        sum(loss).backward()
        optimizer.step()
        # print('global w', R.global_gmm.w)
    i += 1
    print(i)
