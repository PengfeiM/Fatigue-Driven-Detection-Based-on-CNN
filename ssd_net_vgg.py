import torch
import torch.nn as nn
import l2norm
import Config as config
class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.vgg = []
        #vgg-16模型
        self.vgg.append(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1))#conv1_1
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1))#conv1_2
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2,stride=2))#maxpool1
        self.vgg.append(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))#conv2_1
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1))#conv2_2
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2,stride=2))#maxpool2
        self.vgg.append(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1))#conv3_1
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))#conv3_2
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))#conv3_3
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True))#maxpool3
        self.vgg.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))#conv4_1
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))#conv4_2
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))#conv4_3
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2,stride=2))#maxpool4
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))#conv5_1
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))#conv5_2
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))#conv5_3
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=3,stride=1,padding=1))#maxpool5
        self.vgg.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=6,dilation=6))#conv6
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1))#conv7
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg = nn.ModuleList(self.vgg)
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        #特征层位置输出
        self.feature_map_loc_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        #特征层类别输出
        self.feature_map_conf_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*config.class_num,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*config.class_num,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*config.class_num,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*config.class_num,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*config.class_num,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*config.class_num,kernel_size=3,stride=1,padding=1)
        )


    #正向传播过程
    def forward(self, image):
        out = self.vgg[0](image)
        out = self.vgg[1](out)
        out = self.vgg[2](out)
        out = self.vgg[3](out)
        out = self.vgg[4](out)
        out = self.vgg[5](out)
        out = self.vgg[6](out)
        out = self.vgg[7](out)
        out = self.vgg[8](out)
        out = self.vgg[9](out)
        out = self.vgg[10](out)
        out = self.vgg[11](out)
        out = self.vgg[12](out)
        out = self.vgg[13](out)
        out = self.vgg[14](out)
        out = self.vgg[15](out)
        out = self.vgg[16](out)
        out = self.vgg[17](out)
        out = self.vgg[18](out)
        out = self.vgg[19](out)
        out = self.vgg[20](out)
        out = self.vgg[21](out)
        out = self.vgg[22](out)
        my_L2Norm = l2norm.L2Norm(512, 20)
        feature_map_1 = out
        feature_map_1 = my_L2Norm(feature_map_1)
        loc_1 = self.feature_map_loc_1(feature_map_1).permute((0,2,3,1)).contiguous()
        conf_1 = self.feature_map_conf_1(feature_map_1).permute((0,2,3,1)).contiguous()
        out = self.vgg[23](out)
        out = self.vgg[24](out)
        out = self.vgg[25](out)
        out = self.vgg[26](out)
        out = self.vgg[27](out)
        out = self.vgg[28](out)
        out = self.vgg[29](out)
        out = self.vgg[30](out)
        out = self.vgg[31](out)
        out = self.vgg[32](out)
        out = self.vgg[33](out)
        out = self.vgg[34](out)
        feature_map_2 = out
        loc_2 = self.feature_map_loc_2(feature_map_2).permute((0,2,3,1)).contiguous()
        conf_2 = self.feature_map_conf_2(feature_map_2).permute((0,2,3,1)).contiguous()
        out = self.conv8_1(out)
        out = self.conv8_2(out)
        feature_map_3 = out
        loc_3 = self.feature_map_loc_3(feature_map_3).permute((0,2,3,1)).contiguous()
        conf_3 = self.feature_map_conf_3(feature_map_3).permute((0,2,3,1)).contiguous()
        out = self.conv9_1(out)
        out = self.conv9_2(out)
        feature_map_4 = out
        loc_4 = self.feature_map_loc_4(feature_map_4).permute((0,2,3,1)).contiguous()
        conf_4 = self.feature_map_conf_4(feature_map_4).permute((0,2,3,1)).contiguous()
        out = self.conv10_1(out)
        out = self.conv10_2(out)
        feature_map_5 = out
        loc_5 = self.feature_map_loc_5(feature_map_5).permute((0,2,3,1)).contiguous()
        conf_5 = self.feature_map_conf_5(feature_map_5).permute((0,2,3,1)).contiguous()
        out = self.conv11_1(out)
        out = self.conv11_2(out)
        feature_map_6 = out
        loc_6 = self.feature_map_loc_6(feature_map_6).permute((0,2,3,1)).contiguous()
        conf_6 = self.feature_map_conf_6(feature_map_6).permute((0,2,3,1)).contiguous()
        loc_list = [loc_1,loc_2,loc_3,loc_4,loc_5,loc_6]
        conf_list = [conf_1,conf_2,conf_3,conf_4,conf_5,conf_6]
        return loc_list,conf_list
