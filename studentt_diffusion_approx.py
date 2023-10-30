import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import scipy.special as sps
from cycler import cycler

monochrome = (
    cycler('color', ['k']) * cycler('marker', ['', '.']) *
    cycler('linestyle', ['-', '--', ':', '-.'])
)
plt.rc('axes', prop_cycle=monochrome)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 18,
})

def student_t(x, nu):
    return sps.gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * sps.gamma(nu / 2)
                                      ) * (1 + x**2 / nu)**(-(nu + 1) / 2)


def student_gauss_approx(x, nu, N, sigma):
    f1 = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))
    f2 = sps.gamma((nu + 1) / 2
                   ) / ((2 * sigma**2 / nu)**(nu / 2) * sps.gamma(nu / 2))
    fac = f1 * f2
    accum = np.zeros_like(x)
    for n in range(N):
        accum += (x**2 / (2 * sigma**2)
                  )**n * sps.hyperu((nu + 1) / 2, nu / 2 + 1 - n, nu /
                                    (2 * sigma**2)) / sps.factorial(n)
    return fac * accum


def student_gauss_numerical(x, nu, sigma_):
    sigma = sigma_ / (2 * x_max) * N_disc
    gauss_kernel = ss.windows.gaussian(x.shape[0], sigma)
    fx = student_t(x, nu)
    fyt = ss.convolve(fx, gauss_kernel, mode='same')
    dx = x[1] - x[0]
    return fyt / fyt.sum() / dx


N_disc = 10_000
x_max = 20
x = np.linspace(-x_max, x_max, N_disc, dtype=np.float64)
nu = 1
fx = student_t(x, nu)
fx[fx < 1e-3] = np.nan

for t in [.1, 1, 3]:
    sigma = np.sqrt(2 * t)
    fig, ax = plt.subplots()
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    numerical = student_gauss_numerical(x, nu, sigma)
    numerical[numerical < 1e-3] = np.nan
    ax.plot(x, -np.log(fx))
    ax.plot(x, -np.log(numerical))
    Ns = [10, 20, 50, 100]
    for N in Ns:
        approx = student_gauss_approx(x, nu, N, sigma)
        approx[approx < 1e-3] = np.nan
        ax.plot(x, -np.log(approx), markevery=200)
    if t == .1:
        ax.legend(['\\( f_X \\)', '\\( f_{{Y_t}} \\)'] + [f'\\( f_{{Y_t}}^{{{N}}} \\)' for N in Ns])
    plt.savefig(f'./out/studentt_diffusion_approx_{t:.1f}.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
