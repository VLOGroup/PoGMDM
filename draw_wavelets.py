import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch as th

import models

sigmas = [0, .025, .05, .1, .2]
N = len(sigmas)
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.coolwarm(np.linspace(0, 1, N))
)
plt.rcParams.update({
    "text.usetex": True,
})
gamma = .5
diag_mul = .7
n_w = 125
mus_l1 = th.linspace(-gamma, gamma, n_w, device='cuda')[None].repeat(2, 1)
mus_l1d = th.linspace(-gamma * diag_mul, gamma * diag_mul, n_w,
                      device='cuda')[None]
mus_l2 = th.linspace(-gamma * 2, gamma * 2, n_w,
                     device='cuda')[None].repeat(2, 1)
mus_l2d = th.linspace(
    -gamma * 2 * diag_mul, gamma * 2 * diag_mul, n_w, device='cuda'
)[None]
mus = th.cat((mus_l1, mus_l1d, mus_l2, mus_l2d), dim=0)

n_points = 20
ylimss_f = [[-.2, 7.2], [-.2, 7.2]]
levels = 2

for wave, ylims_f in zip(['db2', 'db4'], ylimss_f):
    R = models.WaveletGMM(
        levels=2,
        mus=mus,
        vmin=-.7,
        vmax=.7,
        n_w=n_w,
        w_init='student-t',
        im_sz=64,
        wave=wave
    ).cuda()
    R.set_sigma(0)
    R.load_state_dict(th.load(f'./out/wavelets/{wave}/state_final.pth'))

    xs_pot = []
    for level in range(levels):
        for direction in range(3):
            qs = R.mus[level * 3 + direction].amax()
            xs_pot.append(
                th.linspace(
                    -1.1 * qs,
                    1.1 * qs,
                    n_points**2,
                    device='cuda',
                    dtype=th.float32
                )
            )

    def get_vis(scale=1.):
        weights = R.w
        # weights = th.nn.functional.softmax(weights, dim=1)
        with th.no_grad():
            x = th.linspace(
                scale * R.vmin,
                scale * R.vmax,
                n_points**2,
                dtype=R.w.dtype,
                device=R.w[0].device,
            )[None].repeat(R.n_f, 1)
            pot, act = logsumexp.pot_act(
                x.view(1, R.n_f, n_points, n_points), weights, R.mus, R.sigma
            )
            pot = pot.view(R.n_f, n_points * n_points)
            pot -= pot.amin(dim=1, keepdim=True)
            act = act.view(R.n_f, n_points * n_points)
        return x[0], pot, act

    ylims_p = [-40, 40]
    fig_f, ax_f = plt.subplots(2, 3)
    fig_fp, ax_fp = plt.subplots(2, 3)
    fig_tweedie, ax_tweedie = plt.subplots(2, 3)
    for s in sigmas:
        with th.no_grad():
            R.set_sigma(s)
            for level in range(2):
                for direction in range(3):
                    idx_flat = level * 3 + direction

                    x = xs_pot[idx_flat]
                    pot, act = logsumexp.pot_act(
                        x.view(1, 1, n_points, n_points),
                        R.w.get()[idx_flat:idx_flat + 1], R.mus[idx_flat],
                        R.sigma[idx_flat:idx_flat + 1]
                    )
                    pot = pot.view(n_points * n_points)
                    pot -= pot.amin()
                    pot[pot > 7] = np.nan
                    act = act.view(n_points * n_points)
                    if s == 0:
                        act[:] = np.nan
                    ax_f[level,
                         direction].plot(x.cpu().numpy(),
                                         pot.cpu().numpy())
                    ax_fp[level,
                          direction].plot(x.cpu().numpy(),
                                          act.cpu().numpy())
                    ax_tweedie[level, direction].plot(
                        x.cpu().numpy(),
                        x.cpu().numpy() - s**2 * act.cpu().numpy()
                    )

                    ax_tweedie[level, direction].grid(True)
                    for axx, ylims in zip([ax_f, ax_fp], [ylims_f, ylims_p]):
                        axx[level, direction].set_ylim(ylims)
                        axx[level, direction].grid(True)
                        # if (level, direction) == (1, 0):
                        #     xt = axx[level, direction].get_xticklabels()
                        if (level, direction) != (1, 0):
                            #     axx[level, direction].tick_params(tick1On=False)
                            #     axx[level, direction].set_xticklabels([])
                            axx[level, direction].set_yticklabels([])
                        #     axx[level, direction].set_frame_on(False)
    for fig, name in zip([fig_f, fig_fp, fig_tweedie],
                         ['pot', 'act', 'tweedie']):
        fig.tight_layout()

    h = R.h.detach()
    g = R.get_highpass(h)
    h_list = h.cpu().numpy().tolist()
    g_list = g.cpu().numpy().tolist()

    plt.rcParams.update({
        'font.size': 20,
    })
    wavelet = pywt.Wavelet(f'{wave}learned', [h_list, g_list, h_list, g_list])
    phi_d, psi_d, phi_r, psi_r, x = wavelet.wavefun(6)
    plt.figure()
    plt.plot(x, np.flip(phi_d), 'k--')
    plt.plot(x, np.flip(psi_d), 'k-')
    plt.legend(['$\\phi$', '$\\omega$'])
    plt.title('Learned')
    plt.figure()
    plt.stem(np.flip(h_list), basefmt=' ', linefmt='k', markerfmt='ko')
    plt.title('Learned')

    wavelet = pywt.Wavelet(wave)
    h_ = wavelet.dec_lo
    phi, psi, x = wavelet.wavefun(6)
    plt.figure()
    plt.plot(x, phi, 'k--')
    plt.plot(x, psi, 'k-')
    plt.legend(['$\\phi$', '$\\omega$'])
    plt.title(f'\\texttt{{{wave}}}')
    plt.figure()
    plt.stem(h_, basefmt=' ', linefmt='k', markerfmt='ko')
    plt.title(f'\\texttt{{{wave}}}')

    plt.rcParams.update({
        'font.size': 10,
    })
plt.show()
