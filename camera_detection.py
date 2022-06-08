from torch.autograd import Variable
from detection import *
from ssd_net_vgg import *
from voc0712 import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import utils
import torch.backends.cudnn as cudnn
import time
#检测cuda是否可用
if torch.cuda.is_available():
	print('-----gpu mode-----')
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	print('-----cpu mode-----')
colors_tableau=[ (214, 39, 40),(23, 190, 207),(188, 189, 34)]

net=SSD()
net=torch.nn.DataParallel(net)
net.train(mode=False)
net.load_state_dict(torch.load('./weights/ssd_voc_5000.pth',map_location=lambda storage,loc: storage))
if torch.cuda.is_available():
	net = net.cuda()
	cudnn.benchmark = True

devkit_path='./dataset/'
ftest=open(devkit_path+'ImageSets/Main/test.txt','r')
img_mean=(104.0,117.0,123.0)

cap=cv2.VideoCapture(0)
max_fps=0

while(True):
	flag=True
	num_rec=0
	start=time.time()
	ret,img=cap.read()
	
	x=cv2.resize(img,(300,300)).astype(np.float32)
	x-=img_mean
	x=x.astype(np.float32)
	x=x[:,:,::-1].copy()
	x=torch.from_numpy(x).permute(2,0,1)
	xx=Variable(x.unsqueeze(0))
	if torch.cuda.is_available():
		xx=xx.cuda()
	y=net(xx)
	softmax=nn.Softmax(dim=-1)
	detect=Detect(config.class_num,0,200,0.01,0.45)
	priors=utils.default_prior_box()

	loc,conf=y
	loc=torch.cat([o.view(o.size(0),-1)for o in loc],1)
	conf=torch.cat([o.view(o.size(0),-1)for o in conf],1)
	
	detections=detect(
		loc.view(loc.size(0),-1,4),
		softmax(conf.view(conf.size(0),-1,config.class_num)),
		torch.cat([o.view(-1,4) for o in priors],0)
	).data
	labels=VOC_CLASSES
	top_k=10
	
	scale=torch.Tensor(img.shape[1::-1]).repeat(2)
	for i in range(detections.size(1)):
		
		j=0
		while detections[0,i,j,0]>=0.4:
			score=detections[0,i,j,0]
			label_name=labels[i-1]
			if label_name=='closed_eye':
				flag=False
			display_txt='%s:%.2f'%(label_name,score)
			pt=(detections[0,i,j,1:]*scale).cpu().numpy()
			coords=(pt[0],pt[1]),pt[2]-pt[0]+1,pt[3]-pt[1]+1
			color=colors_tableau[i]
			cv2.rectangle(img,(pt[0],pt[1]),(pt[2],pt[3]),color,2)
			cv2.putText(img,display_txt,(int(pt[0]),int(pt[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
			j+=1
			num_rec+=1
	#cv2.imwrite('./tested/'+name+'.jpg',img)
	#print('Pic:'+name+" writed!")
	if num_rec>0:
		if flag:
			print(' 1:eye-open')
		else:
			print(' 0:eye-closed')
	else:
		print('-1:eye-not detected')
	
	T=time.time()-start
	fps=1/T
	if fps>max_fps:
		max_fps=fps
	fps_txt='fps:%.2f'%(fps)
	cv2.putText(img,fps_txt,(0,10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
	cv2.imshow("ssd",img)
	if cv2.waitKey(100) & 0xff == ord('q'):
		break
#print("-------end-------")
cap.release()
cv2.destroyAllWindows()
print(max_fps)