# 更新/Update
> [!ATTENTION] 这个项目我很久没更新了，由于一些原因吧。最近终于有时间把自己的开发环境重新整好了。  
> 很高兴看到还有人在关注这个项目，只有由于一些个人原因没能及时回复大家的疑问，请谅解。
> 接下来一段时间，我会在工作之余，尽可能修复这个项目中的一些问题（主要是版本不兼容），同时我也会把我自己的项目环境放在
> 代码仓中，如果你使用 miniconda 或者 anaconda，那么你可以直接从 `environment.yml` 导入项目的虚拟环境。
> 即使不适用 conda，我想这个清单也可以在一定程度上帮助你配置环境。
> 暂时我会使用 cpu 先调试项目，可以的话，后续会把 gpu 相关环境配置和方法放上来。

**我的硬件配置**
> 作为参考
```bash
        _,met$$$$$gg.          revan_m@Ebon-Hawk
     ,g$$$$$$$$$$$$$$$P.       -----------------
   ,g$$P""       """Y$$.".     OS: Debian GNU/Linux 12 (bookworm) x86_64
  ,$$P'              `$$$.     Host: Windows Subsystem for Linux - Debian (2.4.13)
',$$P       ,ggs.     `$$b:    Kernel: Linux 5.15.167.4-microsoft-standard-WSL2
`d$$'     ,$P"'   .    $$$     
 $$P      d$'     ,    $$P     
 $$:      $$.   -    ,d$$'     Shell: zsh 5.9
 $$;      Y$b._   _,d$P'       WM: WSLg 1.0.65 (Wayland)
 Y$$.    `.`"Y$$$$P"'          Terminal: tmux 3.5a
 `$$b      "-.__               CPU: 11th Gen Intel(R) Core(TM) i7-11800H (4) @ 2.30 GHz
  `Y$$b                        GPU 1: Microsoft Basic Render Driver
   `Y$$.                       GPU 2: Microsoft Basic Render Driver
     `$$b.                     Memory: 740.60 MiB / 7.63 GiB (9%)
       `Y$$b.                  Swap: 0 B / 2.00 GiB (0%)
         `"Y$b._               
             `""""             
                               
                               
                               
                               
                               Battery (Microsoft Hyper-V Virtual Batte): 100% [AC Connected]
                               Locale: zh_CN.UTF-8
```
啊，看起来显卡信息这里没有。我的机器其实有一个独显3060laptop，相信知道这个显卡的大概了解其性能，不再多做赘述。


# 郑重声明：
本项目是我在github（国内的话是gitee）的免费开源项目。我没有授权任何平台（CSDN、淘宝）付费提供该项目。  
This project is open-source on github and gitee.  
No authorization to any platform to sell my project.  

> [!IMPORTANT] 如果你需要论文，请直接发[邮件](PengfeiM@outlook.com)，不要提 issue，issue用来解决问题，或者提出你的想法。

## 更新说明
针对近来很多同学反映的在新版本pytorch下程序报错：
"RuntimeError: Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method. (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)" 。
做了修改了detection、Test、xxx_detection等部分代码，可适应新版本pytorch。另外旧版本pytor请选择old-pytorch分支。


## 运行环境（Excution Environment）：

> 1.python 3.7.1  
> 2.pytorch 1.0.1  
> 3.python-opencv  
> 4.cuda大概可能是8或者9，时间太久记不清了。   不过主要还是显卡-cuda-cudnn-pytorch版本对应吧。

## 说明（Notions）

预训练的权重文件[vgg_16]

1、具体的配置文件请看 Config.py 文件--file that save the configuration    
2、训练运行 python Train.py        --file that start the training and control the loops  
3、单张测试 python test.py         --file that test ssd with one image  
4、测试网络性能 python eval.py     --file that evaluate the performance  
5、测试视频 python camera_detection.py --file that test the cnn with a video sequence  

## 目前进度（Process: All Done）：

| 内容             | 进度 |
| ---------------- | ---- |
| PERCLOS 计算     | DONE |
| 眨眼频率计算     | DONE |
| 打哈欠检测及计算 | DONE |
| 疲劳检测         | DONE |

## 主要文件说明（File in the repo）：

ssd_net_vgg.py 定义 class SSD 的文件（define the ssd cnn）  
Train.py 训练代码  (training)  
voc0712.py 数据集处理代码（没有改文件名，改的话还要改其他代码，麻烦）  (processing the dataset)  
loss_function.py 损失函数  (loss function)  
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
（针对部分代码中涉及的文件（指ssd_voc_5000_plus.pth），翻了翻旧U盘，算是找到了。）
百度云：
[数据集和权重文件](https://pan.baidu.com/s/1cgl94gxSNEW0ZI-wYcZtpQ)
提取码：hwsi  
Onedrive：
[数据集](https://mailustceducn-my.sharepoint.com/:u:/g/personal/mpf916_mail_ustc_edu_cn/ER0UB-cAe1VDp9hJZ7e5Ef4B7kGvVX4PePSj7WRtb9VrLQ?e=lbDnjV)
[权重文件](https://mailustceducn-my.sharepoint.com/:f:/g/personal/mpf916_mail_ustc_edu_cn/EqGCPA3SGz5Mp-RMHJSoSSwBg-KG09qwgSAPiOjMOcVVtQ?e=v5yhQz)

## 测试

1、运行 Train.py 训练
2、eval 可以用于测试整个测试集，test 用于单张图片测试。

## 关于问题讨论
欢迎大家就代码中存在的问题提issue，同时本存储库开放了讨论功能（Discussion），欢迎各位将一些共性的问题放到Dicussion中提问（我也会将部分以前的issue放到Discussion中）。

## 关于咨询
如果issue和Discussion不能满足您的需要，随时可以发邮件到我的邮箱(PengfeiM@outlook.com)提出您的问题。
当然，不管是issue/discussion还是邮件，我都会尽快回复（issue和discussion有更新github会给我发邮件，我也会时常检查github手机端APP）。

**最后，如果您想要支持我的工作，请扫描下面的二维码**
![我的支付宝](https://user-images.githubusercontent.com/45191163/116050673-55db0400-a6aa-11eb-9588-cc0546e89f70.jpg)

**谢谢您对我的支持和帮助**
