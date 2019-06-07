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
import torch.utils.data as  data
import sys
import os
import pickle

#检测cuda是否可用
if torch.cuda.is_available():
	print('-----gpu mode-----')
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	print('-----cpu mode-----')
colors_tableau=[ (214, 39, 40),(23, 190, 207),(188, 189, 34),(188,34,188),(205,108,8)]

net=SSD()
net=torch.nn.DataParallel(net)
net.train(mode=False)
net.load_state_dict(torch.load('./weights/ssd300_voc_100000.pth',map_location=lambda storage,loc: storage))
if torch.cuda.is_available():
	net = net.cuda()
	cudnn.benchmark = True

devkit_path='./dataset/'
annopath=os.path.join(devkit_path,'Annotations', '%s.xml')
ftest=open(devkit_path+'ImageSets/Main/test.txt','r')
img_mean=(104.0,117.0,123.0)

def parse_rec(filename):
	'''获取图片中所有的label和坐标'''
	tree=ET.parse(filename)
	objects=[]
	for obj in tree.findall('object'):
		obj_struct={}
		obj_struct['name']=obj.find('name').text
		bbox=obj.find('bndbox')
		obj_struct['bbox']=[int(bbox.find('xmin').text)-1,
							int(bbox.find('ymin').text)-1,
							int(bbox.find('xmax').text)-1,
							int(bbox.find('ymax').text)-1]
		objects.append(obj_struct)
		
	return objects

def IoU(obj_R,obj_P):
	#计算交并比
	cood_r=obj_R['bbox']
	cood_p=obj_P['bbox']
	ixmin=max(cood_r[0],cood_p[0])
	iymin=max(cood_r[1],cood_p[1])
	ixmax=min(cood_r[2],cood_p[2])
	iymax=min(cood_r[3],cood_p[3])
	iw=max(ixmax-ixmin,0.)
	ih=max(iymax-iymin,0.)
	inters=iw*ih*1.0
	uni=((cood_r[2]-cood_r[0])*(cood_r[3]-cood_r[1])+
	     (cood_p[2]-cood_p[0])*(cood_p[3]-cood_p[1])-
		 inters)
	overlaps=inters/uni
	return overlaps
	
count=0
time_start=time.time()
accu_num=0
real_num=0

for line in ftest:
	name=line.strip()
	print(name)
	obj_real=parse_rec(devkit_path+'Annotations/'+name+'.xml')
	real_num+=len(obj_real)
	img=cv2.imread(devkit_path+'JPEGImages/'+name+'.jpg',cv2.IMREAD_COLOR)
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
	obj_pre=[]
	for i in range(detections.size(1)):
		j=0
		
		while detections[0,i,j,0]>=0.4:
			score=detections[0,i,j,0]
			obj={}
			obj['name']=labels[i-1]
			pt=(detections[0,i,j,1:]*scale).cpu().numpy()
			obj['bbox']=[int(pt[0]),
					     int(pt[1]),
						 int(pt[2]),
						 int(pt[3])]
			obj_pre.append(obj)
		
			label_name=labels[i-1]
			display_txt='%s:%.2f'%(label_name,score)
			coords=(pt[0],pt[1]),pt[2]-pt[0]+1,pt[3]-pt[1]+1
			color=colors_tableau[i]
			cv2.rectangle(img,(pt[0],pt[1]),(pt[2],pt[3]),color,2)
			cv2.putText(img,display_txt,(int(pt[0]),int(pt[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
		
			j+=1
			
	#把测试过的图片写入磁盘
	#cv2.imwrite('./tested/'+name+'.jpg',img)
	#print('Pic:'+name+" writed!")
	
	for obj_R in obj_real:
		for obj_P in obj_pre:
			if IoU(obj_R,obj_P)>0.5:#阈值暂设为0.5
				if obj_R['name']==obj_P['name']:
					accu_num+=1
	count+=1
print("-------end-------")
elapsed=(time.time()-time_start)
print('共{:d}张图片\n用时：{:f} s\nfps={:f}\n准确率：{:f}'
	  .format(count,elapsed,count/elapsed,accu_num/real_num))