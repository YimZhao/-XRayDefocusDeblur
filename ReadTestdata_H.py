import cv2 as cv
import numpy as np
import math
from scipy.io import loadmat
from prop_class_asm import propagate

datapath = './conditions/condition1/recon_34733_128_t2_object_ave.npy'
#data = np.load(datapath)
data_ori = np.load(datapath)
data = data_ori
clim_phase = [-0.4, 0.07]
phase= np.angle(data)
amp=np.abs(data)
#phase= cv.resize(np.angle(data), (512,512),interpolation=cv.INTER_CUBIC)
print(phase.max(),phase.min())
phase = 1-(phase-phase.min())/(phase.max()-phase.min())
cv.imshow('phase', phase)
#cv.imshow('phase0', 1-phase)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('./conditions/condition1/test/recon_34733_128_t2_object_ave.png',phase*255)