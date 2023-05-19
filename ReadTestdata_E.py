import cv2 as cv
import numpy as np
import math
from scipy.io import loadmat
from prop_class_asm import propagate

datapath = './conditions/condition2/offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.mat'
#data = np.load(datapath)
data_ori = loadmat(datapath)
data = data_ori['object']
obj_layer1 = data_ori['object_layer'][0][0]
obj_layer2 = data_ori['object_layer'][0][1]
'''
m,n = data.shape
phase = np.zeros(data.shape)
for i in range(m):
    for j in range(n):
        x = np.real(data[i,j])
        y = np.imag(data[i,j])
        if y < 0:
            phase[i,j] = -np.arccos(x/np.sqrt(x*x+y*y))
        elif y > 0:
            phase[i,j] = np.arccos(x/np.sqrt(x*x+y*y))
        else:
            if x > 0:
                phase[i,j] = 0
            if x <= 0:
                phase[i,j] = -math.pi
#phase = (phase+math.pi)/(2*math.pi)
'''
clim_phase = [-1, 0.3]
phase= cv.resize(np.angle(data), (512,512),interpolation=cv.INTER_CUBIC)
phase = 1-(phase-clim_phase[0])/(clim_phase[1]-clim_phase[0])
#phase = (phase-phase.min())/(phase.max()-phase.min())
#print(phase.shape)
phase_lay1 = cv.resize(np.angle(obj_layer1[0,0]), (512,512),interpolation=cv.INTER_CUBIC)
phase_lay1 = 1-(phase_lay1-clim_phase[0])/(clim_phase[1]-clim_phase[0])
phase_lay2 = cv.resize(np.angle(obj_layer2[0,0]), (512,512),interpolation=cv.INTER_CUBIC)
phase_lay2 = 1-(phase_lay2-clim_phase[0])/(clim_phase[1]-clim_phase[0])
cv.imshow('phase', phase)
#cv.imshow('phase0', 1-phase)
cv.imshow('layer1', phase_lay1)
cv.imshow('layer2', phase_lay2)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.imwrite('./conditions/condition2/test/offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.png',phase*255)
#cv.imwrite('./conditions/condition2/test/lay1_offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.png',phase_lay1*255)
#cv.imwrite('./conditions/condition2/test/lay2_offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.png',phase_lay2*255)

pixel_size = 43.9e-9
wvlength = 0.2e-9
sep = 200e-6
lay2 = propagate(phase_lay2,1,pixel_size,pixel_size,wvlength,sep)
lay2 = abs(lay2)
res = cv.add(lay2,phase_lay1)
cv.imshow('prop', res)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.imwrite('./conditions/condition2/test/prop/prop'+str(sep)+'_offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.png',res*255)