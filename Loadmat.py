#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: etsai
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


### Load reconstruction
fn = './conditions/condition2/offline_201511_case2_S00459_192x192_3DM_it1000_recons_19.mat'
recon = loadmat(fn)
#recon = loadmat('offline_201511_case2_S00459_192x192_3DM_it1000_recons_19.mat')

obj = recon['object'] # Combined layers

print(obj.shape)
obj_layer1 = recon['object_layer'][0][0]
obj_layer2 = recon['object_layer'][0][1]



### Plot

plt.figure(1); plt.clf()
M, N = [2,3]; fig=1
clim_phase = [-1, 0.3]
clim_amp = [0.7, 1.2]

plt.subplot(M, N, fig); plt.title(fn+'\nobject phase')
plt.imshow(np.angle(obj), cmap='bone', clim=clim_phase, origin='lower'); plt.colorbar() 
fig = fig+1

plt.subplot(M, N, fig); plt.title('object layer1 phase')
plt.imshow(np.angle(obj_layer1[0][0]), cmap='bone', clim=clim_phase, origin='lower'); plt.colorbar()
fig = fig+1

plt.subplot(M, N, fig); plt.title('object layer2 phase')
plt.imshow(np.angle(obj_layer2[0][0]), cmap='bone', clim=clim_phase, origin='lower'); plt.colorbar()
fig = fig+1


plt.subplot(M, N, fig); plt.title('object amp')
plt.imshow(np.abs(obj), cmap='bone', clim=clim_amp); plt.colorbar()
fig = fig+1

plt.subplot(M, N, fig); plt.title('object layer1 amp')
plt.imshow(np.abs(obj_layer1[0][0]), cmap='bone', clim=clim_amp, origin='lower'); plt.colorbar()
fig = fig+1

plt.subplot(M, N, fig); plt.title('object layer2 amp')
plt.imshow(np.abs(obj_layer2[0][0]), cmap='bone', clim=clim_amp, origin='lower'); plt.colorbar()
fig = fig+1

plt.show()