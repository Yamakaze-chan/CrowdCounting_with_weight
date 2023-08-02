# =============================================================================
# original package
# =============================================================================
import torch
import numpy as np
from torch import optim
from pathlib import Path
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# =============================================================================
# creatived package
# =============================================================================
from model import LibraNet, weights_normal_init
from buffer import ReplayBuffer
from train_test import train_model,test_model

''' 3 DIFFERENT METHODS TO REMEMBER:
 - torch.save(arg, PATH) # can be model, tensor, or dictionary
 - torch.load(PATH)
 - torch.load_state_dict(arg)
'''

''' 2 DIFFERENT WAYS OF SAVING
# 1) lazy way: save whole model
torch.save(model, PATH)
# model class must be defined somewhere
model = torch.load(PATH)
model.eval()
# 2) recommended way: save only the state_dict
torch.save(model.state_dict(), PATH)
# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''
# Parameters
# =============================================================================   
parameters = {'TRAIN_SKIP':100,
             'BUFFER_LENGTH':10000,
             'ERROR_RANGE':0.5,
             'GAMMA':0.9,
             'batch_size':128,
             'Interval_N':57,
             'step_log':0.1,
             'start_log':-2,
             'HV_NUMBER':8,
             'ACTION_NUMBER':9,
             'ERROR_SYSTEM':0,
             'means':[[108.25673428], [ 97.02240046], [ 93.37483706]]}

###########load checkpoint#####################

net = LibraNet(parameters) 

#learning_rate = 0.01

#FILE = "model_ckpt.pth.tar"

#checkpoint = torch.load(FILE)
#net.load_state_dict(checkpoint['state_dict'])

#epoch = checkpoint['epoch']

print("Load check point model!")
net.load_state_dict(torch.load('model_ckpt.pth.tar')['state_dict'])
net.cuda()
net.eval() 

# Create the preprocessing transformation here
toTensor = transforms.ToTensor()
means = torch.FloatTensor(np.array(parameters['means']) / 255).unsqueeze(0).unsqueeze(2).cuda()

#test_path ='data/Test/' 
#test_img = test_path + 'images/'
#test_gt = test_path + 'ground_truth/'

#gt_path = test_gt + 'GT_IMG_1.mat'  
#gt = sio.loadmat(gt_path)
#print(gt)         

#gt = sio.loadmat("C:\\Users\\ACER\AI\\libranet\\data\\Test\\ground_truth\\GT_IMG_141.mat") #SanghaiTech part A
gt = sio.loadmat("C:\\Users\\ACER\\AI\\libranet\\test_pics\\archive\\ShanghaiTech\\part_B\\test_data\\ground-truth\\GT_IMG_137.mat")
print(len(gt['image_info'][0][0][0][0][0]))   
#img_name = test_img+ 'IMG_1.jpg'      
#img_name =  "C:\\Users\\ACER\\AI\\libranet\\test_pics\\archive\\ShanghaiTech\\part_B\\test_data\\images\\IMG_137.jpg"
img_name =  "c:\\Users\\ACER\\AI\\libranet\\test_pics\\343.jpg"
Img = cv2.imread(img_name)

h = Img.shape[0]
w = Img.shape[1]
                    
#gt = len(gt['image_info'][0][0][0][0][0])

ht = int(32*int(h/32))    
wt = int(32*int(w/32))
if ht != h:
    ht = int(32 * (int(h / 32) + 1))  
if wt != w:
    wt = int(32 * (int(w / 32) + 1))  
    
ho = int(ht/32)
wo = int(wt/32)
                                                
Img_t = np.zeros((ht, wt,3))
Img_t[0:h, 0:w, :] = Img.copy()
Img = Img_t.astype(np.uint8)
        
img = toTensor(Img).unsqueeze(0).cuda()-means
                                            
featuremap_t = []        
class_rem = np.zeros((ho, wo))          
hv_save = np.zeros((ho, wo, parameters['HV_NUMBER']))
        
mask_last = np.zeros((ho, wo))
mask_last = mask_last.astype(np.int8)

Q_val = []
act = []

featuremap_t = net.get_feature(im_data=img)
for step in range(0, parameters['HV_NUMBER']): 
    
    hv = torch.from_numpy(hv_save.transpose((2, 0, 1))).unsqueeze_(0).float().cuda()
    
    Q = net.get_Q(feature=featuremap_t, history_vectory=hv)
                
    Q = -Q[0].data.cpu().numpy()  
    sort = Q.argsort(axis=0)
    #print(Q)
    
    action_max = np.zeros((ho, wo))
    
    mask_max_find = np.zeros((ho,wo))
    for recycle_ind in range(0,parameters['ACTION_NUMBER']):
        maskselect_end = (sort[recycle_ind] == parameters['ACTION_NUMBER']-1)
        action_sort = sort[recycle_ind]

        #use for 32x32

        
        #print(action_sort)

        A_sort = np.squeeze(net.A_mat[action_sort])
        print(A_sort) #Action

        if(Q[recycle_ind].flatten().tolist() != [8]):
            Q_val.append(Q[recycle_ind].flatten().tolist())
        else: 
            Q_val.append([-1])
        if(A_sort.tolist() !=999):
            act.append(A_sort.tolist())
        else:
            act.append(11)

        
        _ind_max = (((class_rem + A_sort <  parameters['Interval_N']) & (class_rem +A_sort >= 0) | maskselect_end) & ( mask_max_find == 0)) & (mask_last == 0)
        action_max[_ind_max] = action_max[_ind_max] + sort[recycle_ind] [_ind_max]
        #print(action_max[_ind_max])
        mask_max_find = mask_max_find + ((class_rem + A_sort <  parameters['Interval_N']) & (class_rem +A_sort >= 0) | maskselect_end).astype(np.int8)
        #print(mask_max_find)

    
    #plot 32x32
    #print(act)
    k = -1
    act, Q_val = zip(*sorted(zip(act, Q_val)))
    ind_act = act.index(11)-1
    plt.scatter(act[ind_act], Q_val[ind_act], c=act.index(11), cmap='hot')
    #plt.scatter(act[act.index(11)-1], Q_val[act.index(11)-1], c=act.index(11), cmap='hot')
    print(Q_val)
    print(act)
    print("sum of act: " + str(sum(act,-11)))
    plt.plot(act, Q_val, label = "t="+str(step))
    Q_val = []
    act = []

    
        
    
    mask_select_end=(action_max == parameters['ACTION_NUMBER']-1).astype(np.int8)
    class_rem = class_rem + (1 - mask_select_end) * (1 - mask_last) * (np.squeeze(net.A_mat_h_w[action_max.astype(np.int8)]))
    hv_save[:, :, step] = action_max+1 
    mask_now = mask_last.copy()
    mask_now = mask_now | mask_select_end
    mask_last = mask_now.copy()
    if (1 - mask_last).sum() == 0:
        
        #use for 32x32

        # naming the x axis
        plt.xlabel('Action')
        # naming the y axis
        plt.ylabel('Q_value')
        # show a legend on the plot
        plt.legend()
        # function to show the plot
        plt.show()

        break         

count_rem = net.class2num[class_rem.astype(np.int8)]
#print(count_rem)
est = count_rem.sum()
#print("Ground truth is " + str(gt))
string_txt = "Estimate is " + str(est)
print(string_txt)
"""
# Window name in which image is displayed
window_name = 'Image'
  
# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (0, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
   
# Using cv2.putText() method
image = cv2.putText(Img, string_txt, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
   
# Displaying the image
cv2.imshow(window_name, image) 
cv2.waitKey()
"""