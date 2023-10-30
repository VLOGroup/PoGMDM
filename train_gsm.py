import time

import numpy as np
import torch as th
import torchvision.utils as tvu
from torch.utils.tensorboard.writer import SummaryWriter

import data
import models
import optim
import util
import vis

bs = 7000
kernel_size = 7
n_f = kernel_size**2 - 1
patch_size = kernel_size
color = False
rotate = True
flip = True
log_freq = 200
n_scales = 20

dataset = data.BSDS(color, bs, patch_size, rotate, flip)
R = models.ProductGSM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    ortho=True,
    n_scales=n_scales,
    kernel_size=kernel_size,
    K_init='random',
).cuda()

R.set_sigma(0)
ipalm = False

t = str(time.time())
writer = SummaryWriter(log_dir='./log/' + t)
lrs = {
    'w': 1e-5,
    'K.weight': 1e-2,
    'sigmas_0': 3e-4,
}
if ipalm:
    groups = []
    for k, v in R.named_parameters():
        groups.append({
            'params': v,
            'name': k,
        })
    optimizer = optim.IPalm(groups, eps=0)
else:
    groups = []
    for k, v in R.named_parameters():
        if k not in lrs.keys():
            print(k)
            continue
        groups.append({
            'params': v,
            'lr': lrs[k],
            'name': k,
        })
    optimizer = optim.AdaBelief(groups)

div = th.ones_like(
    util.patch2image(
        util.image2patch(
            th.ones((patch_size, patch_size)).cuda(),
            (kernel_size, kernel_size)
        ), (patch_size, patch_size), (kernel_size, kernel_size)
    )
)


def loss_criterion(y, R, sigmas):
    n = [th.randn_like(y) for _ in range(len(sigmas))]

    def closure(compute_grad=False):
        with th.set_grad_enabled(compute_grad):
            loss_sm = 0
            for sigma, noise in zip(sigmas, n):
                R.set_sigma(sigma)
                x = y + sigma * noise
                loss_sm += ((sigma * R.grad(x)[1] / div - noise)**2).sum()
            return [loss_sm / bs]

    return closure


for i, y in enumerate(dataset):
    closure = loss_criterion(y, R, np.random.rand(10) * .4)

    writer.add_scalar('loss/score', sum(closure()).item(), global_step=i)

    if i % log_freq == 0:
        writer.add_figure('theta', vis.vis_gms(R), i)
        for s in [5 / 255, 15 / 255, 25 / 255, 50 / 255]:
            with th.no_grad():
                R.set_sigma(s)
                x = y[:25] + s * th.randn_like(y[:25])
                y_hat = x - R.grad(x)[1] / div * s**2
                stack = th.concat((x[:8], y_hat[:8], y[:8]), dim=0)
                writer.add_scalar(
                    f'psnr {int(s * 255)}', psnr(y[:25], y_hat), global_step=i
                )
                writer.add_image(
                    f'test {int(s * 255)}',
                    tvu.make_grid(th.clip(stack, 0, 1), nrow=8),
                    global_step=i
                )
                R.set_sigma(0)

        th.save(R.state_dict(), f'./out/gsm/state_{i:06d}.pth')

    if ipalm:
        loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        loss = closure(True)
        sum(loss).backward()
        if any([th.isnan(p.grad).any() for p in R.parameters()]):
            continue
        optimizer.step()
