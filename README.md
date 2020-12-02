## 运行环境：

> 1.python 3.7.1
> 2.pytorch 1.0.1
> 3.python-opencv

## 说明

预训练的权重文件[vgg_16]

1、具体的配置文件请看 Config.py 文件  
2、训练运行 python Train.py  
3、单张测试 python test.py  
4、测试网络性能 python eval.py  
5、测试视频 python camera_detection.py

## 目前进度：

| 内容             | 进度 |
| ---------------- | ---- |
| PERCLOS 计算     | DONE |
| 眨眼频率计算     | DONE |
| 打哈欠检测及计算 | DONE |
| 疲劳检测         | DONE |

## 主要文件说明：

ssd_net_vgg.py 定义 class SSD 的文件  
Train.py 训练代码  
voc0712.py 数据集处理代码（没有改文件名，改的话还要改其他代码，麻烦）  
loss_function.py 损失函数  
detection.py 检测结果的处理代码，将 SSD 返回结果处理为 opencv 可以处理的形式  
eval.py 评估网络性能代码  
test.py 单张图片测试代码 Ps:没写参数接口，所以要改测试的图片就要手动改代码内部文件名了  
l2norm.py l2 正则化  
Config.py 配置参数  
utils.py 工具类  
camera.py opencv 调用摄像头测试  
camera_detection.py 摄像头检测代码 V1,V2  
video_detection.py 视频检测，V3

## 数据集结构：

> /dataset:
>
> > /Annotations 存放含有目标信息的 xml 文件  
> > /ImageSets/Main 存放图片名的文件  
> > /JPEGImages 存放图片  
> > /gray2rgb.m 灰度图转三通道  
> > /txt.py 生成 ImageSets 文件的代码

## 权重文件存放路径：

weights
测试后的图片存放位置：
tested

## 参考代码：

https://github.com/amdegroot/ssd.pytorch

## 数据集和权重文件：

链接：https://pan.baidu.com/s/1c6UZeEioruy2SzGkSKDWPQ
提取码：euci

## 测试

1、运行 Train.py 训练
2、eval 可以用于测试整个测试集，test 用于单张图片测试。
