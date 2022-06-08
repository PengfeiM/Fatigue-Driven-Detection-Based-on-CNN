# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:58:49 2019

@author: 朋飞
"""

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
colors_tableau=[ (214, 39, 40),(23, 190, 207),(188, 189, 34),(188,34,188),(205,108,8)]

def Yawn(list_Y,list_y1):
	list_cmp=list_Y[:len(list_Y1)]==list_Y1
	for flag in list_cmp:
		if flag==False:
			return False
	return True
#初始化网络
net=SSD()
net=torch.nn.DataParallel(net)
net.train(mode=False)
net.load_state_dict(torch.load('./weights/ssd300_VOC_100000.pth',map_location=lambda storage,loc: storage))
if torch.cuda.is_available():
	net = net.cuda()
	cudnn.benchmark = True

img_mean=(104.0,117.0,123.0)

#打开视频文件，file_name改成0即为打开摄像头
file_name='C:/Users/HP/Desktop/9-FemaleNoGlasses.avi'
cap=cv2.VideoCapture(file_name)
max_fps=0

#保存检测结果的List
#眼睛和嘴巴都是，张开为‘1’，闭合为‘0’
video_fps=20#视频fps=20
list_B=np.ones(video_fps*3)#眼睛状态List,建议根据fps修改，视频fps=20
list_Y=np.zeros(video_fps*10)#嘴巴状态list，10s
list_Y1=np.ones(int(video_fps*1.5))#如果在list_Y中存在list_Y1，则判定一次打哈欠(大约1.5s)，
list_Y1[int(video_fps*1.5)-1]=0#从持续张嘴到闭嘴判定为一次打哈欠
list_blink=np.ones(video_fps*10)#大约是记录10S内信息，眨眼为‘1’，不眨眼为‘0’
list_yawn=np.zeros(video_fps*30)#大约是半分钟内打哈欠记录，打哈欠为‘1’，不打哈欠为‘0’

#blink_count=0#眨眼计数
#yawn_count=0
#blink_start=time.time()#炸眼时间
#yawn_start=time.time()#打哈欠时间
blink_freq=0.5
yawn_freq=0
#开始检测，按‘q’退出
while cap.isOpened():
	flag_B=True#是否闭眼的flag
	flag_Y=False#张嘴flag

	num_rec=0#检测到的眼睛的数量
	start=time.time()#计时
	ret,img=cap.read()#读取图片
	
	#检测
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
	# detect=Detect(config.class_num,0,200,0.01,0.45)
	detect = Detect.apply
	priors=utils.default_prior_box()

	loc,conf=y
	loc=torch.cat([o.view(o.size(0),-1)for o in loc],1)
	conf=torch.cat([o.view(o.size(0),-1)for o in conf],1)
	
	detections=detect(
		loc.view(loc.size(0),-1,4),
		softmax(conf.view(conf.size(0),-1,config.class_num)),
		torch.cat([o.view(-1,4) for o in priors],0),
		config.class_num,
		200,
    	0.7,
    	0.45
	).data
	labels=VOC_CLASSES
	top_k=10
	
	#将检测结果放置于图片上
	scale=torch.Tensor(img.shape[1::-1]).repeat(2)
	for i in range(detections.size(1)):
		
		j=0
		while detections[0,i,j,0]>=0.4:
			score=detections[0,i,j,0]
			label_name=labels[i-1]
			if label_name=='closed_eye':
				flag_B=False
			if label_name=='open_mouth':
				flag_Y=True
			display_txt='%s:%.2f'%(label_name,score)
			pt=(detections[0,i,j,1:]*scale).cpu().numpy()
			coords=(pt[0],pt[1]),pt[2]-pt[0]+1,pt[3]-pt[1]+1
			color=colors_tableau[i]
			cv2.rectangle(img,(pt[0],pt[1]),(pt[2],pt[3]),color,2)
			cv2.putText(img,display_txt,(int(pt[0]),int(pt[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
			j+=1
			num_rec+=1
	if num_rec>0:
		if flag_B:
			#print(' 1:eye-open')
			list_B=np.append(list_B,1)#睁眼为‘1’
		else:
			#print(' 0:eye-closed')
			list_B=np.append(list_B,0)#闭眼为‘0’
		list_B=np.delete(list_B,0)
		if flag_Y:
			list_Y=np.append(list_Y,1)
		else:
			list_Y=np.append(list_Y,0)
		list_Y=np.delete(list_Y,0)
	else:
		print('nothing detected')
	#print(list)
	
	if list_B[13]==1 and list_B[14]==0:
		#如果上一帧为’1‘，此帧为’0‘则判定为眨眼
		print('----------------眨眼----------------------')
		list_blink=np.append(list_blink,1)
	else:
		list_blink=np.append(list_blink,0)
	list_blink=np.delete(list_blink,0)
	
	
	#检测打哈欠
	#if Yawn(list_Y,list_Y1):
	if (list_Y[len(list_Y)-len(list_Y1):]==list_Y1).all():
		print('----------------------打哈欠----------------------')
		list_Y=np.zeros(50)#此处是检测到一次打哈欠之后将嘴部状态list全部置‘0’，考虑到打哈欠所用时间较长，所以基本不会出现漏检
		list_yawn=np.append(list_yawn,1)
	else:
		list_yawn=np.append(list_yawn,0)
	list_yawn=np.delete(list_yawn,0)
	
	
	
	#实时计算PERCLOS perblink,peryawn
	#即计算平均闭眼时长百分比，平均眨眼百分比，平均打哈欠百分比
	perclos=1-np.average(list_B)
	perblink=np.average(list_blink)
	peryawn=np.average(list_yawn)
	#print('perclos={:f}'.format(perclos))
	
	#此处为判断疲劳部分
	#想法1：两个频率计算改为实时的，所以此处不再修改
	if(perclos>0.4 or perblink<2.5/(10*video_fps) or peryawn>3/(30*video_fps)):
		print('疲劳')
	else:
		print('清醒')
	
	
	T=time.time()-start
	fps=1/T#实时在视频上显示fps
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
#print(max_fps)