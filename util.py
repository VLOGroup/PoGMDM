import math
import unittest

import pytorch_wavelets as pwt
import torch as th
import torch.nn.functional as F
import torch.nn.functional as tf

import models
import shltutils.filters as filters


def get_models(sz=320):
    M, N = sz, sz
    n_w = 63 * 2 - 1
    kernel_size = 7
    n_f = kernel_size**2 - 1
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
    ).cuda()
    R_gmm.load_state_dict(th.load('./out/patch/state_final.pth'))
    n_scales = 20
    R_gsm = models.ProductGSM(
        n_f=n_f,
        bound_norm=False,
        zero_mean=True,
        ortho=True,
        n_scales=n_scales,
        kernel_size=kernel_size,
        K_init='random',
    ).cuda()
    R_gsm.load_state_dict(th.load('./out/gsm/state_final.pth'))

    kernel_size = 15
    n_f = kernel_size**2 - 1
    R_gmm15 = models.ProductGMM(
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
    ).cuda()
    R_gmm15.load_state_dict(th.load('./out/patch15/state_final.pth'))
    n_w = 125

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
    R_sh = models.GMMConv(
        n_scales=2,
        symmetric=True,
        vmin=-.5,
        vmax=.5,
        n_w=n_w,
        w_init='abs',
        lamda=lamda.cuda(),
        h=h.cuda(),
        dims=(sz, sz),
        P=P.cuda()
    ).cuda()
    R_sh.load_state_dict(th.load('./out/shearlets/state_final.pth'))
    R_sh.set_eta(shape=(sz, sz))
    R_sh.set_sigma(0)

    n_w = 125
    Rs_wavelet = []
    for wave in ['db2', 'db4']:
        levels = 2
        K = pwt.DWTForward(wave=wave, J=levels,
                           mode='reflect').cuda().to(th.float64)
        mus_each = []
        xs_pot = []
        data = th.load('./celeba-resized.pth').cuda().to(th.float64)[:5000]
        n_points = 20
        for level in range(levels):
            for direction in range(3):
                qs = th.quantile(
                    K(data[:10_000])[1][level][:, :, direction], 0.999
                ) * 1.1
                # qs = .7
                mus_each.append(th.linspace(-qs, qs, n_w, device='cuda')[None])
                xs_pot.append(
                    th.linspace(
                        -qs, qs, n_points**2, device='cuda', dtype=th.float64
                    )
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
            im_sz=sz,
            wave=wave,
        ).cuda().to(th.float64)
        R.set_sigma(0)
        R.load_state_dict(th.load(f'./out/wavelets/{wave}/state_final.pth'))
        Rs_wavelet.append(R.to(th.float32))
    return R_gmm, R_gmm15, R_gsm, *Rs_wavelet, R_sh


def mahalanobis(x, mus, precs_chol):
    distances = th.empty((mus.shape[0], x.shape[0]), device=x.device)
    for k, (mu, L) in enumerate(zip(mus, precs_chol)):
        distances[k] = (((x - mu) @ L)**2).sum(1)
    return distances


def lognormal(x, mus, prec):
    # print(th.linalg.slogdet(prec)[0])
    prec_chol = th.linalg.cholesky(prec)
    maha_dist = mahalanobis(x, mus, prec_chol)
    logdets = th.linalg.slogdet(prec_chol)[1]
    return -(math.log(2 * math.pi) * mus.shape[1] +
             maha_dist) / 2 + logdets[:, None]


def patch2image(batches, img_dims, size=(2, 2), stride=(1, 1)):
    tmp = batches[None].permute(0, 2, 1)
    return th.squeeze(F.fold(tmp, img_dims, size, 1, 0, stride))


def image2patch(image, size=(2, 2), stride=(1, 1)):
    return image.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    ).contiguous().view(-1, size[0] * size[1])


def proj_simplex(
    x: th.Tensor,  # 1-D vector of weights
    s: float = 1.,  # axis intersection
):
    k = th.linspace(1, len(x), len(x), device=x.device)
    x_s = th.sort(x, dim=0, descending=True)[0]
    t = (th.cumsum(x_s, dim=0) - s) / k
    mu = th.max(t)
    return th.clamp(x - mu, 0, s)


def proj_simplex_simul(
    x: th.Tensor,  # 2-D array of weights,
    # projection is performed along the last axis axis
    s: float = 1.,  # axis interesection
):
    K = x.shape[1]
    k = th.linspace(1, K, K, device=x.device)
    x_s = th.sort(x, dim=1, descending=True)[0]
    t = (th.cumsum(x_s, dim=1) - s) / k[None]
    mu = th.max(t, dim=1, keepdim=True)[0]
    return th.clamp(x - mu, 0, s)


def weight_init(
    vmin: float,
    vmax: float,
    n_w: int,
    scale: float,
    mode: str,
) -> th.Tensor:
    x = th.linspace(vmin, vmax, n_w, dtype=th.float32)
    match mode:
        case "constant":
            w = th.ones_like(x) * scale
        case "linear":
            w = x * scale
        case "quadratic":
            w = x**2 * scale
        case "abs":
            w = th.abs(x) * scale
            w -= w.max()
            w = w.abs()
        case "student-t":
            alpha = 100
            w = scale * math.sqrt(alpha) / (1 + 0.5 * alpha * x**2)
        case "Student-T":
            a_ = 0.1 * 78
            b_ = 0.1 * 78**2
            denom = 1 + (a_ * x)**2
            w = b_ / (2 * a_**2) * th.log(denom)

    return w


def _get_gauss_kernel() -> th.Tensor:
    return th.tensor([[
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]]) / 256.0


def _compute_padding(kernel_size: list[int]) -> list[int]:
    computed = [k // 2 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def psnr(x, y):
    return th.mean(20 * th.log10(1 / th.sqrt(th.mean((x - y)**2,
                                                     (1, 2, 3))))).item()


def init_params(mode, vmin, vmax, n_f, n_w, symmetric):
    x = th.linspace(vmin, vmax, n_w, dtype=th.float32)
    match mode:
        case "constant":
            w = th.ones_like(x)
        case "linear":
            w = x
        case "quadratic":
            w = -x**2
            w -= w.min()
        case "abs":
            w = -th.abs(x)
            w -= w.min()
        case "student-t":
            alpha = 1000
            w = math.sqrt(alpha) / (1 + 0.5 * alpha * x**2)
        case "Student-T":
            a_ = 0.1 * 78
            b_ = 0.1 * 78**2
            denom = 1 + (a_ * x)**2
            w = b_ / (2 * a_**2) * th.log(denom)
        case 'random':
            w = th.rand((n_w, ))
    w /= w.sum()

    if symmetric:
        w = w[w.shape[0] // 2:][None].repeat(n_f, 1).clone()
    else:
        w = th.nn.Parameter(w[None].repeat(n_f, 1).clone())
    return w


class SSIM(th.nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            "w",
            th.ones(1, 1, win_size, win_size) / win_size**2
        )
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
        self.w: th.Tensor

    def forward(
        self,
        X: th.Tensor,
        Y: th.Tensor,
        data_range: th.Tensor | float = 1.,
        reduced: bool = True,
    ):
        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2
        ux = tf.conv2d(X, self.w)
        uy = tf.conv2d(Y, self.w)
        uxx = tf.conv2d(X * X, self.w)
        uyy = tf.conv2d(Y * Y, self.w)
        uxy = tf.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return S.mean()
        else:
            return S
