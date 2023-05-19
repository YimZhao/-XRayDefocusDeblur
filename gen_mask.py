from prop_class_asm import propagate
import cv2 
import numpy as np
import os
import random

imgpath = "E:/Thesis/BNL/Deblurring/MyCode/data/Cell/"
datasetpath = "E:/Thesis/BNL/Deblurring/MyCode/Cell_gen/v5/"
datasetname = "cond1_256real"
imgsavepath = datasetpath+datasetname+"/input/"
gtsavepath = datasetpath+datasetname+"/gt/"
masksavepath = datasetpath+datasetname+"/mask/"
imgfiles = os.listdir(imgpath)
bgfiles = random.sample(imgfiles,len(imgfiles))

#print(bgfiles)
for file, bgfile in zip(imgfiles,bgfiles):
    if not os.path.isdir(file):
        imgsize = (256,256)
        resultsize = (256,256,3)
        img = cv2.imread(imgpath+"/"+file)
        img = cv2.resize(img,imgsize)
        bg = cv2.imread(imgpath+'/'+ bgfile)
        bg = cv2.resize(bg,imgsize)
        imggray = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2GRAY)
        bggray = cv2.cvtColor(cv2.bitwise_not(bg), cv2.COLOR_BGR2GRAY)
        
        #array, direction, x_pixel_size_m, y_pixel_size_m, wavelength_m, z_m
        #cond1
        pixel_size = 7.33783143939e-9
        wvlength = 0.103333e-9
        sep = 10e-6
        '''
        #cond2
        pixel_size = 43.9e-9
        wvlength = 0.2e-9
        sep = 200e-6
        '''
        p = propagate(bggray,1,pixel_size,pixel_size,wvlength,sep)
        #bg_p = abs(p)
        #imggray blur for real condition
        #img_p = propagate(imggray,1,pixel_size,pixel_size,wvlength,np.random.normal(0,2)*1e-6)
        #img_p = abs(img_p)
        #cv2.imshow("bg",bggray) x 
       
        gt = np.zeros(resultsize)
        #overlay mask
        mask = np.zeros(imgsize)
        result = np.zeros(resultsize)
        '''
        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                result[i,j]= max(img_p[i,j],bg_p[i,j])
        '''
        result = p*imggray/255
        result = abs(result)
        result = 255-result
        # cv2.imshow('p',abs(p/255))
        # cv2.imshow('b',imggray)
        # cv2.imshow('r',result/255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        gt[:,:,2] = cv2.bitwise_not(bggray)
        gt[:,:,1] = cv2.bitwise_not(imggray)
        # cv2.imshow("gt",gt)
        # cv2.imshow("bg",bggray)
        # cv2.imshow("img",imggray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        for i in range(imgsize[0]):
            for j in range(imgsize[1]):
                if abs(p[i,j])<250 and imggray[i,j]<250:
                    mask[i,j] = 255
                else:
                    mask[i,j] = 0
        #print(result)
        '''
        cv2.imshow("ori",imggray)
        cv2.imshow("prop",result)
        cv2.imshow("gt", gt)
        cv2.imshow("mask",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        print(gt.max())
        '''
        cv2.imwrite(imgsavepath+'('+str(sep)+')_'+file[0:-4]+bgfile[0:-4]+".png",result)
        cv2.imwrite(gtsavepath+'('+str(sep)+')_'+file[0:-4]+bgfile[0:-4]+".png",gt)
        cv2.imwrite(masksavepath+'('+str(sep)+')_'+file[0:-4]+bgfile[0:-4]+".png",mask)
