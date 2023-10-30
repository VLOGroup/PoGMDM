import time

import numpy as np
import torch as th
import torchvision.utils as tvu
from torch.utils.tensorboard.writer import SummaryWriter

import data
import models
import optim
import shltutils.filters as filters
import util

bs = 16
patch_size = 96
color = False
rotate = True
flip = True
log_freq = 20
n_w = 125

dataset = data.BSDS(color, bs, patch_size, rotate, flip)

h = th.tensor([
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
],
              device='cuda')
lamda = th.ones((21, ), dtype=th.float32, device='cuda')
lamda = lamda.float()
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = th.from_numpy(filters.modulate2(h0, 'c')).float()
R = models.GMMConv(
    n_scales=2,
    symmetric=True,
    vmin=-.5,
    vmax=.5,
    n_w=n_w,
    w_init='student-t',
    lamda=lamda.cuda(),
    h=h.cuda(),
    dims=(patch_size, patch_size),
    P=P.cuda()
).cuda()
R.set_eta()
R.set_sigma(0)
ipalm = False

t = str(time.time())
writer = SummaryWriter(log_dir='./log/shearlets/' + t)
lrs = {
    'w.w': 1e-4,
    'lamda': 5e-5,
    'h': 5e-5,
    'P': 5e-5,
}
if ipalm:
    groups = []
    for k, v in R.named_parameters():
        groups.append({
            'params': v,
            'name': k,
        })
    optimizer = optim.IPalm(groups, eps=1e-5)
else:
    groups = []
    for k, v in R.named_parameters():
        # if k not in lrs.keys():
        # print(k)
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
            return [loss_sm / bs]

    return closure


for i, y in enumerate(dataset):
    closure = loss_criterion(y, R, np.random.rand(10) * .4)

    if i % log_freq == 0:
        writer.add_scalar('loss/score', sum(closure()).item(), global_step=i)
        #     writer.add_figure('theta', vis.vis(R), i)

        for s in [0.025, 0.05, 0.1, 0.2]:
            with th.no_grad():
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

    th.save(R.state_dict(), f'./out/shearlets/state_{i:06d}.pth')

    if ipalm:
        loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        loss = closure(True)
        sum(loss).backward()
        optimizer.step()
    # Update eta after each parameter update
    R.set_eta()
