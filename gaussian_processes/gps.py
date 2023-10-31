import gsum as gm
import numpy as np
from scikit.
kernel = RBF(length_scale=ls, length_scale_bounds='fixed') + \
    WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
gp = gm.ConjugateGaussianProcess(kernel=kernel, center=center, df=np.inf, scale=sd, nugget=0)

trunc_gp = gm.TruncationGP(kernel=kernel, ref=ref, ratio=ratio, disp=0, df=np.inf, scale=1, optimizer=None)
trunc_gp.fit