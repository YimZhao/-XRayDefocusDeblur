import cv2 as cv
import torch
from masknet import DefocusMaskNet
import os
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from dataset import MyCustomDatasetTest
import DRBNet
import numpy as np

norm_mean = (.5,.5,.5)
norm_std = (.5,.5,.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasetname = 'test1'
ckptname = 'rnd_pyramid'
testpath = './Cell_gen/'+datasetname
testpath = './conditions/condition1/test/'
#testpath = './Cell_gen/v5/cond1val_256real/input/'
#testgt = './Cell_gen/val/'+datasetname+'/gt'
imgs = os.listdir(testpath)
val_data = MyCustomDatasetTest(testpath,transform)
val_loader = torch.utils.data.DataLoader(val_data,batch_size=1,shuffle=False)
criterion = nn.L1Loss()

print("*")
lossbook_i = []
lossbook_o = []
for i,data in enumerate(val_loader,0):
    img = data
    img = Variable(img.to(device))
    #gt = Variable(gt.to(device))
    
    with torch.no_grad():
        Net = DRBNet.DRBNet_single().to(device)
        #ckptpath = './deblur_mask_ckpt/'+ckptname+'/cell_model_e48.pkl'
        ckptpath = './deblur_mask_ckpt/cond1_256/e47.pkl'
        Net.load_state_dict(torch.load(ckptpath))
        output = Net(img)
        #loss_i = criterion(img,gt)
        #loss_o = criterion(output,gt)
        #print(i,loss_i.data.cpu().numpy(),loss_o.data.cpu().numpy())
        #lossbook_o.append(loss_o.data.cpu().numpy())
        #lossbook_i.append(loss_i.data.cpu().numpy())
    output_cpu = (output.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    input_cpu = (img.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    #gt_cpu = (gt.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0
    #cv.imwrite("./result/"+ckptname+'/'+datasetname+"/"+imgs[i],output_cpu*255)
    cv.imwrite('./result/cond1/256real/'+imgs[i], output_cpu*255)
#np.savetxt('./result/'+ckptname+'/'+datasetname+'/outputloss.txt',lossbook_o)
#np.savetxt('./result/'+ckptname+'/'+datasetname+'/inputloss.txt',lossbook_i)
#mean = []
#mean.append(np.mean(lossbook_i))
#mean.append(np.mean(lossbook_o))
#np.savetxt('./result/'+ckptname+'/'+datasetname+'/mean.txt',mean)