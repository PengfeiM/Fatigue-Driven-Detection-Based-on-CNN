import torch
# import pdb

from torch.autograd import Variable
from detection import Detect
# from ssd_net_vgg import *
from ssd_net_vgg import SSD
# from voc0712 import *
from voc0712 import VOC_CLASSES
import Config as config
import torch.nn as nn
import numpy as np
import cv2
import utils
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229), (158, 218, 229), (158, 218, 229)]

net = SSD()    # initialize SSD
net = torch.nn.DataParallel(net)
net.train(mode=False)
# net.load_state_dict(torch.load('./weights/ssd300_VOC_100000.pth',map_location=lambda storage, loc: storage))
net.load_state_dict(torch.load('./weights/ssd_voc_120000.pth', map_location=lambda storage, loc: storage))
img_id = 60
name = 'test'
image = cv2.imread('./' + name + '.jpg', cv2.IMREAD_COLOR)
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)
softmax = nn.Softmax(dim=-1)
# detect = Detect(config.class_num, 0, 200, 0.01, 0.45)
detect = Detect.apply  # pytorch新版本需要这样使用
priors = utils.default_prior_box()

loc, conf = y
loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

detections = detect(
    loc.view(loc.size(0), -1, 4),
    softmax(conf.view(conf.size(0), -1, config.class_num)),
    torch.cat([o.view(-1, 4) for o in priors], 0),
    config.class_num,
    200,
    0.7,
    0.45
).data
# detections = detect.apply

labels = VOC_CLASSES
top_k = 10

# plt.imshow(rgb_image)  # plot the image for matplotlib

# scale each detection back up to the image
scale = torch.Tensor(image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0, i, j, 0] >= 0.4:
        score = detections[0, i, j, 0]
        label_name = labels[i - 1]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        rec = [int(e) for e in pt]
        # pdb.set_trace()
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors_tableau[i]
        # cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
        print(rec)
        cv2.rectangle(img=image, rec=rec, color=color, thickness=2)
        cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
        j += 1
# cv2.imshow('test', image)
# cv2.waitKey(100000) # not implemented in headless version
print("------end-------")
cv2.imwrite(name + '_done.jpg', image)
