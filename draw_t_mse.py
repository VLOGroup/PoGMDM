import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as tf

import data
import models

plt.rcParams.update({
    "text.usetex": True,
})

n_f = 48
bs = 64 * 4_000
patch_size = 7
kernel_size = 7
color = False
rotate = True
flip = True
log_freq = 50
n_w = 63 * 2 - 1

dataset = data.BSDS(color, bs, patch_size, rotate, flip)
sigmas = th.linspace(0, 0.2, 50).cuda()
R = models.ProductGMM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    symmetric=False,
    ortho=True,
    vmin=-1,
    vmax=1,
    kernel_size=kernel_size,
    K_init='random',
    n_w=n_w,
    w_init='student-t',
    sigmas=sigmas,
).cuda()
th.set_grad_enabled(False)

R.set_sigma(0)
R._loss_sigmas = sigmas

fig, ax = plt.subplots(figsize=(10, 3))
for dash, ckpt in zip(
    ['-', '--'],
    ['./patch_final/state.pth', './patch_5/state_004300.pth'],
):
    state = R.load_state_dict(th.load(ckpt), strict=False)

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
        print(y.shape)
        closure = loss_criterion(y, R, sigmas.tolist())
        closure()
        break
    fs = []

    for sig in sigmas:
        R.set_sigma(sig.item())
        K, (x, f), (x_hist, hist, _), = R.get_vis()
        fs.append(f)

    fs = th.stack(fs).permute(1, 0, 2)
    hist = hist.permute(1, 0, 2)
    chist = th.cumsum(hist, -1)
    chist = chist / chist.max(-1, keepdim=True)[0]
    hist = -th.log(hist)
    hist -= hist.min(-1, keepdim=True)[0]
    fs -= fs.min(-1, keepdim=True)[0]
    fs = tf.interpolate(fs, (125, ))
    for color, q in zip(
        ['tab:blue', 'tab:orange', 'tab:green'],
        [.005, .01, .02],
    ):
        i_low = (chist - q).abs().argmin(-1)
        i_high = (chist - (1 - q)).abs().argmin(-1)
        res = th.zeros(sigmas.shape[0]).cuda()
        for j, (low, high) in enumerate(zip(i_low, i_high)):
            for s, (l, h) in enumerate(zip(low.tolist(), high.tolist())):
                res[s] += (((fs[j, s, l:h] - hist[j, s, l:h])**2) /
                           fs[j, s, l:h].max()**2).sum() / (h - l)

        ax.plot(sigmas.tolist(), res.cpu(), color=color, linestyle=dash)

ax.set_ylim([-.01, .25])
ax.plot([0.02, 0.02], [-.01, .25], 'k-')
ax.set_xlabel('\\( \\sqrt{2t} \\)')
ax.set_ylabel('\\( \\mathrm{NMSE}_{\\kappa} \\)')
plt.grid('on')
plt.tight_layout()
plt.show()
