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
net.load_state_dict(torch.load('./weights/ssd_voc_5000_plus.pth',map_location=lambda storage,loc: storage))
if torch.cuda.is_available():
	net = net.cuda()
	cudnn.benchmark = True

img_mean=(104.0,117.0,123.0)

#调用摄像头
cap=cv2.VideoCapture(0)
max_fps=0

#保存检测结果的List
#眼睛和嘴巴都是，张开为‘1’，闭合为‘0’
list_B=np.ones(15)#眼睛状态List,建议根据fps修改
list_Y=np.zeros(50)#嘴巴状态list，建议根据fps修改
list_Y1=np.ones(5)#如果在list_Y中存在list_Y1，则判定一次打哈欠，同上，长度建议修改
blink_count=0#眨眼计数
yawn_count=0
blink_start=time.time()#炸眼时间
yawn_start=time.time()#打哈欠时间
blink_freq=0.5
yawn_freq=0
#开始检测，按‘q’退出
while(True):
	flag_B=True#是否闭眼的flag
	flag_Y=False
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
	#实时计算PERCLOS
	perclos=1-np.average(list_B)
	print('perclos={:f}'.format(perclos))
	if list_B[13]==1 and list_B[14]==0:
		#如果上一帧为’1‘，此帧为’0‘则判定为眨眼
		print('----------------眨眼----------------------')
		blink_count+=1
	blink_T=time.time()-blink_start
	if blink_T>10:
		#每10秒计算一次眨眼频率
		blink_freq=blink_count/blink_T
		blink_start=time.time()
		blink_count=0
		print('blink_freq={:f}'.format(blink_freq))
	#检测打哈欠
	#if Yawn(list_Y,list_Y1):
	if (list_Y[len(list_Y)-len(list_Y1):]==list_Y1).all():
		print('----------------------打哈欠----------------------')
		yawn_count+=1
		list_Y=np.zeros(50)
	#计算打哈欠频率
	yawn_T=time.time()-yawn_start
	if yawn_T>60:
		yawn_freq=yawn_count/yawn_T
		yawn_start=time.time()
		yawn_count=0
		print('yawn_freq={:f}'.fomat(yawn_freq))
		
	#此处为判断疲劳部分
	'''
	想法1：最简单，但是太影响实时性
	if(perclos>0.4 or blink_freq<0.25 or yawn_freq>5/60):
		print('疲劳')
		if(blink_freq<0.25)
	else:
		print('清醒')
	'''
	#想法2：
	if(perclos>0.4):
		print('疲劳')
	elif(blink_freq<0.25):
		print('疲劳')
		blink_freq=0.5#如果因为眨眼频率判断疲劳，则初始化眨眼频率
	elif(yawn_freq>5.0/60):
		print("疲劳")
		yawn_freq=0#初始化，同上
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