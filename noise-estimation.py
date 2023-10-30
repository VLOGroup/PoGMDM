import matplotlib.pyplot as plt
import numpy as np
import torch as th
from cycler import cycler

import data
import models

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

n_f = 48
bs = 1000
patch_size = 7
kernel_size = 7
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
    vmin=-1,
    vmax=1,
    kernel_size=kernel_size,
    K_init='random',
    n_w=n_w,
    w_init='student-t',
).cuda()
th.set_grad_enabled(False)

R.load_state_dict(th.load('./out/patch/state_final.pth'))

for y in dataset:
    break

sigmas = np.linspace(0, .5, 200)
sigmas_est = np.linspace(0, 1, 200)
n = th.randn_like(y)
energy = np.empty((sigmas.shape[0], sigmas_est.shape[0]))

for i_s, sigma in enumerate(sigmas):
    for i_e, sigma_est in enumerate(sigmas_est):
        R.set_sigma(sigma_est)
        energy[i_s, i_e] = R.grad(y + n * sigma)[0].mean().item()
    energy[i_s] -= energy[i_s].min()
# energy -= energy.min() - 10

# energy -= energy.min() - 1
mins = energy.argmin(1)
print(mins.shape)
print(energy[np.arange(energy.shape[0]), mins])

energies_ = energy.copy()
energy[energy > 50] = np.nan

fig, ax = plt.subplots(
    subplot_kw={
        "projection": "3d",
        'elev': 31,
        'azim': 42,
        'box_aspect': (1, 2, 1)
    },
    figsize=(6, 6)
)
X, Y = np.meshgrid(sigmas, sigmas_est)
Z = energy
surf = ax.plot_surface(
    X,
    Y,
    Z.T,
    cmap='coolwarm',
    linewidth=1,
    antialiased=True,
    zorder=1,
    rstride=1,
    cstride=1
)
ax.plot(
    sigmas,
    sigmas_est[mins],
    energy[np.arange(energy.shape[0]), mins],
    zorder=10
)
ax.plot(
    sigmas, sigmas, energy[np.arange(energy.shape[0]), mins] + 1, zorder=10
)
# ax.plot(np.ones_like(sigmas) * .1, np.linspace(0, sigmas_est.max(), len(sigmas)), np.zeros_like(sigmas), 'k-.')
ax.set_proj_type('ortho')
ax.set_yticks([x / 10 for x in range(11)])
ax.set_xlabel('\\( \\sigma \\)')
ax.set_ylabel('\\( \\sqrt{2t} \\)')
ax.set_zlabel(
    '\\( \\mathbb{E}_{p\\sim f_X,\\eta\\sim\\mathcal{N}(0, \\mathrm{Id})} l_\\theta(p + \\sigma\\eta, t) \\)'
)

monochrome = (
    cycler('color', ['k']) * cycler('marker', ['', '.']) *
    cycler('linestyle', ['-', '--', ':', '-.'])
)
# plt.rc('axes', prop_cycle=monochrome)
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, 4))
)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
})
fig, ax = plt.subplots(figsize=(6, 4))
e_01 = np.where(energies_[40] < 80, energies_[40], np.nan)
e_02 = np.where(energies_[80] < 80, energies_[80], np.nan)
e_03 = np.where(energies_[120] < 80, energies_[120], np.nan)
e_04 = np.where(energies_[160] < 80, energies_[160], np.nan)
ax.plot(sigmas_est, e_01, label='\\( \\sigma = 0.1 \\)')
ax.plot(sigmas_est, e_02, label='\\( \\sigma = 0.2 \\)')
ax.plot(sigmas_est, e_03, label='\\( \\sigma = 0.3 \\)')
ax.plot(sigmas_est, e_04, label='\\( \\sigma = 0.4 \\)')
ax.legend()
# ax.plot(sigmas_est, np.zeros_like(sigmas_est), 'k-.')
ax.set_xticks([x / 10 for x in range(11)])
print(sigmas_est[mins][40])
plt.grid('on')
ax.set_xlabel('\\( \\sqrt{2t} \\)')
ax.set_ylabel(
    '\\( \\mathbb{E}_{p\\sim f_X,\\eta\\sim\\mathcal{N}(0, \\mathrm{Id})} l_\\theta(p + \\sigma\\eta, t) \\)'
)
plt.show()
