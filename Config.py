'''
本项目是我在github（国内的话是gitee）的免费开源项目。如果你在某些平台（CSDN、淘宝）付费下载了该项目，烦请告知（邮箱(PengfeiM@outlook.com)）。
'''
import os.path as osp
sk = [ 15, 30, 60, 111, 162, 213, 264 ]
feature_map = [ 38, 19, 10, 5, 3, 1 ]
steps = [ 8, 16, 32, 64, 100, 300 ]
image_size = 300
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
MEANS = (104, 117, 123)
batch_size = 8
data_load_number_worker = 0
lr = 5e-4
momentum = 0.9
weight_decacy = 5e-4
gamma = 0.1
VOC_ROOT = osp.join('./', "dataset/")
dataset_root = VOC_ROOT
use_cuda = True
lr_steps = (80000, 100000, 120000)
max_iter = 120000
class_num = 5
