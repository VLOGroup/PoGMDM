import matplotlib.pyplot as plt
import numpy as np
import torch as th
from cycler import cycler

import data
import models
import shltutils.filters as filters
import shltutils.shearlet as ss

sigmas = [0, .025, .05, .1, .2]
N = len(sigmas)
monochrome = (
    cycler('color', ['k']) * cycler('marker', ['', '.']) *
    cycler('linestyle', ['-', '--', ':', '-.'])
)
# plt.rc('axes', prop_cycle=monochrome)
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, N))
)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
})

bs = 16 * 4
patch_size = 96
color = False
rotate = True
flip = True
n_w = 125

dataset = data.BSDS(color, bs, patch_size, rotate, flip)

h = th.tensor([
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
],
              device='cuda')
h_ = h.clone()
lamda = th.ones((21, ), dtype=th.float32, device='cuda')
lamda = lamda.float()
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = th.from_numpy(filters.modulate2(h0, 'c')).float()
print(P.shape)
gamma = .5
R = models.GMMConv(
    n_scales=2,
    symmetric=True,
    vmin=-gamma,
    vmax=gamma,
    n_w=n_w,
    w_init='abs',
    lamda=lamda.cuda(),
    h=h.cuda(),
    dims=(patch_size, patch_size),
    P=P.cuda()
).cuda()
R.load_state_dict(th.load('./out/shearlets/state_final.pth'))
R.set_eta((96, 96))
plt.rcParams.update({
    'font.size': 30,
})
plt.figure()
plt.imshow(P.cpu().numpy(), cmap='gray')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.figure()
plt.stem(R.h.detach().cpu().numpy(), basefmt=' ', linefmt='k', markerfmt='ko')
plt.tight_layout()
plt.figure()
plt.stem(h_.cpu().numpy(), basefmt=' ', linefmt='k', markerfmt='ko')
plt.tight_layout()
plt.figure()
plt.imshow(R.P.detach().cpu().numpy(), cmap='gray')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.rcParams.update({
    'font.size': 10,
})

plt.figure()
plt.scatter(np.arange(10), R.lamda.detach().cpu().numpy()[:10])
plt.scatter(np.arange(10), R.lamda.detach().cpu().numpy()[10:-1])
print(R.lamda.detach().cpu().numpy()[:-1])
sh = ss.ShearletSystem2D(
    2, (96, 96), h=R.h.detach(), P=R.P.detach(), version=1
)
fig_sh, ax_sh = plt.subplots(2, 5 * 2, figsize=(5 * 2, 2))
fig_shtd, ax_shtd = plt.subplots(2, 5 * 2, figsize=(5 * 2, 2))
fig_pot, ax_pot = plt.subplots(2, 5 * 2, figsize=(5 * 2, 2))
fs = []

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, N))
)


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


for y in dataset:
    break

th.set_grad_enabled(False)
ylim = [-.2, 4]
dm = 0.01
bin_edges = th.linspace(
    -gamma - dm / 2,
    gamma + dm / 2,
    n_w + 1,
).cuda()
x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
hist = th.zeros((len(sigmas), R.K.n_shearlets, n_w)).cuda()
for i_s, sigma in enumerate(sigmas):
    R.set_sigma(sigma)
    Kx = R.K((y + sigma * th.randn_like(y)).squeeze())
    for k in range(R.K.n_shearlets):
        hist[i_s,
             k] = th.histogram(Kx[:, k].reshape(-1).cpu(),
                               bin_edges.cpu())[0].to('cuda')

fs = []
fps = []
scale = 1.1
n_points = 20
x = th.linspace(
    -scale * gamma,
    scale * gamma,
    n_points**2,
    dtype=R.w.w.dtype,
    device=R.w.w.device,
)[None].repeat(R.K.n_shearlets, 1)
n_f = R.K.n_shearlets
for sig in sigmas:
    R.set_sigma(sig)
    f, fp = R.pot_act(x.view(1, n_f, n_points, n_points))
    f = f.view(n_f, n_points * n_points)
    fp = fp.view(n_f, n_points * n_points)
    fs.append(f)
    fps.append(fp)

x = x[0]
for j in range(2):
    for k in range(5):
        for next_cone in [True, False]:
            shearlet = sh.shearlets[j * 5 + k + 10 * next_cone]
            r = j
            c = k + 5 * next_cone
            ax_sh[r, c].imshow(shearlet.abs().cpu().numpy(), cmap='gray')
            ax_sh[r, c].axis('off')

            shtd = crop_center(
                sh.shearlets_td[j * 5 + k + 10 * next_cone].real.cpu().numpy(),
                20, 20
            )
            ax_shtd[r, c].imshow(shtd, cmap='gray')
            ax_shtd[r, c].axis('off')
            for i, sigma in enumerate(sigmas):
                if (r, c) != (1, 0):
                    ax_pot[r, c].tick_params(tick1On=False)
                    ax_pot[r, c].set_xticklabels([])
                    ax_pot[r, c].set_yticklabels([])
                    ax_pot[r, c].set_frame_on(False)
                ax_pot[r, c].set_ylim(ylim)
                pot = fs[i][j * 5 + k + 10 * next_cone].cpu().numpy()
                pot -= pot.min()
                ax_pot[r, c].plot(x.cpu().numpy(), pot)
                ax_pot[r, c].grid(True)

for fig, title in zip([fig_sh, fig_shtd, fig_pot],
                      ['spectrum', 'time', 'potentials']):
    fig.tight_layout()
plt.show()
