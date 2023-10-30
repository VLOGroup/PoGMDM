import matplotlib.pyplot as plt
import numpy as np
import torch as th
import imageio

import shltutils.shearlet as ss
import shltutils.filters as filters

sz = 2000
state = th.load('./out/shearlets/state_final.pth')
h, P = state['h'], state['P']
hs = [h]
ps = [P]

h = th.tensor([
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
],
              device='cuda')
hs.append(h)
lamda = th.ones((21, ), dtype=th.float32, device='cuda')
lamda = lamda.float()
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = th.from_numpy(filters.modulate2(h0, 'c')).float().cuda()
ps.append(P)
next_cone = False
for name, h, P in zip(['learned', 'original'],hs, ps):
    inp = []
    shs = ss.ShearletSystem2D(2, (sz, sz), h=h, P=P, version=1)
    for nc_outer in [True, False]:
        for j in range(2):
            for k in range(5):
                sh = shs.shearlets[j * 5 + k + 10 * nc_outer].abs()
                sh_norm = sh / (sh**2).sum().sqrt()
                inners = []
                for nc_inner in [True, False]:
                    for jj in range(2):
                        for kk in range(5):
                            sh_other = shs.shearlets[jj * 5 + kk +
                                                    10 * nc_inner].abs()
                            sh_other_norm = sh_other / (sh_other**
                                                              2).sum().sqrt()

                            cos_sim = (sh_norm * sh_other_norm).sum().item() ** 2
                            inners.append(cos_sim)
                            # print(f'{:.2f}', end=' ')
                    # print()
                inp.append(inners)

    im = np.array(inp)
    imageio.imsave(f'./out/shearlets/figures/inp_{name}.png', (im * 255).astype(np.uint8))
