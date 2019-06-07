import torch
vgg_weights = torch.load('./vgg16_reducedfc.pth')
print(vgg_weights.keys())