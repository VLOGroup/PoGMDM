import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.utils.tensorboard.writer import SummaryWriter

import data
import models
import shltutils.filters as filters

bs = 16 * 4
patch_size = 96
color = False
rotate = True
flip = True
log_freq = 20
n_w = 63 * 2 - 1

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
h0 /= np.sqrt(2)

P = th.from_numpy(filters.modulate2(h0, 'c')).float()
R = models.GMMConv(
    n_f=lamda.shape[0],
    n_scales=2,
    symmetric=False,
    vmin=-1.5,
    vmax=1.5,
    n_w=n_w,
    w_init='student-t',
    lamda=lamda.cuda(),
    h=h.cuda(),
    dims=[patch_size, patch_size],
    P=P.cuda()
).cuda()
R.set_sigma(0)
shearlets = R.K.shearlets.cpu()
for sh in shearlets:
    plt.figure()
    plt.imshow(th.abs(sh).numpy())
    plt.show()
