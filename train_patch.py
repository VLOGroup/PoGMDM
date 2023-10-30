import time

import numpy as np
import torch as th
import torchvision.utils as tvu
from torch.utils.tensorboard.writer import SummaryWriter

import data
import models
import optim
import util

bs = 1_000
kernel_size = 7
n_f = kernel_size**2 - 1
patch_size = kernel_size
color = False
rotate = True
flip = True
log_freq = 50
n_w = 63 * 2 - 1

dataset = data.BSDS(color, bs, patch_size, rotate, flip)
R = models.ProductGMM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    symmetric=True,
    ortho=True,
    vmin=-1.,
    vmax=1.,
    kernel_size=kernel_size,
    K_init='random',
    n_w=n_w,
    w_init='student-t',
).cuda()
R.set_sigma(0)
ipalm = False

t = str(time.time())
writer = SummaryWriter(log_dir='./log/patch/' + t)
lrs = {
    'w.w': 5e-6,
    'K.weight': 5e-3,
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
        print(k)
        groups.append({
            'params': v,
            'lr': lrs[k],
            'name': k,
        })
    optimizer = optim.AdaBelief(groups)

div = util.patch2image(
    util.image2patch(
        th.ones((patch_size, patch_size)).cuda(), (kernel_size, kernel_size)
    ), (patch_size, patch_size), (kernel_size, kernel_size)
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
    closure = loss_criterion(y, R, np.random.rand(100) * .4)

    writer.add_scalar('loss/score', sum(closure()).item(), global_step=i)
    if i % log_freq == 0:
        for s in [.025, .05, .1, .2]:
            with th.no_grad():
                R.set_sigma(s)
                x = y[:25] + s * th.randn_like(y[:25])
                y_hat = x - R.grad(x)[1] / div * s**2
                stack = th.concat((x[:8], y_hat[:8], y[:8]), dim=0)
                writer.add_scalar(
                    f'psnr {int(s * 255)}',
                    util.psnr(y[:25], y_hat),
                    global_step=i
                )
                writer.add_image(
                    f'test {int(s * 255)}',
                    tvu.make_grid(th.clip(stack, 0, 1), nrow=8),
                    global_step=i
                )
                R.set_sigma(0)

        th.save(R.state_dict(), f'./out/patch/state_{i:06d}.pth')

    if ipalm:
        loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        loss = closure(True)
        sum(loss).backward()
        optimizer.step()
    if i == 100_000:
        break
