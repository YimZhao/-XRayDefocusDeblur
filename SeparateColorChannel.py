import cv2
import os
imgpath = './result/cond2/prop/'
imgname = 'prop0.0002_offline_201511_e_S00459_192x192_3DM_it0_MLit2000_recons.png'
img = cv2.imread(imgpath+imgname)
b,g,r = cv2.split(img)
cv2.imwrite('./separate/'+imgname+'_green.png',g)
cv2.imwrite('./separate/'+imgname+'_red.png',r)