import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import Config

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun,self).__init__()
    def forward(self, prediction,targets,priors_boxes):
        loc_data , conf_data = prediction
        loc_data = torch.cat([o.view(o.size(0),-1,4) for o in loc_data] ,1)
        conf_data = torch.cat([o.view(o.size(0),-1,Config.class_num) for o in conf_data],1)
        priors_boxes = torch.cat([o.view(-1,4) for o in priors_boxes],0)
        if Config.use_cuda:
            loc_data = loc_data.cuda()
            conf_data = conf_data.cuda()
            priors_boxes = priors_boxes.cuda()
        # batch_size
        batch_num = loc_data.size(0)
        # default_box数量
        box_num = loc_data.size(1)
        # 存储targets根据每一个prior_box变换后的数据
        target_loc = torch.Tensor(batch_num,box_num,4)
        target_loc.requires_grad_(requires_grad=False)
        # 存储每一个default_box预测的种类
        target_conf = torch.LongTensor(batch_num,box_num)
        target_conf.requires_grad_(requires_grad=False)
        if Config.use_cuda:
            target_loc = target_loc.cuda()
            target_conf = target_conf.cuda()
        # 因为一次batch可能有多个图，每次循环计算出一个图中的box，即8732个box的loc和conf，存放在target_loc和target_conf中
        for batch_id in range(batch_num):
            target_truths = targets[batch_id][:,:-1].data
            target_labels = targets[batch_id][:,-1].data
            if Config.use_cuda:
                target_truths = target_truths.cuda()
                target_labels = target_labels.cuda()
            # 计算box函数，即公式中loc损失函数的计算公式
            utils.match(0.5,target_truths,priors_boxes,target_labels,target_loc,target_conf,batch_id)
        pos = target_conf > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 相当于论文中L1损失函数乘xij的操作
        pre_loc_xij = loc_data[pos_idx].view(-1,4)
        tar_loc_xij = target_loc[pos_idx].view(-1,4)
        # 将计算好的loc和预测进行smooth_li损失函数
        loss_loc = F.smooth_l1_loss(pre_loc_xij,tar_loc_xij,size_average=False)

        batch_conf = conf_data.view(-1,Config.class_num)

        # 参照论文中conf计算方式，求出ci
        loss_c = utils.log_sum_exp(batch_conf) - batch_conf.gather(1, target_conf.view(-1, 1))

        loss_c = loss_c.view(batch_num, -1)
        # 将正样本设定为0
        loss_c[pos] = 0

        # 将剩下的负样本排序，选出目标数量的负样本
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)

        # 提取出正负样本
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, Config.class_num)
        targets_weighted = target_conf[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum().double()
        loss_l = loss_loc.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
