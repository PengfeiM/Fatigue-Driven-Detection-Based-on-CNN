import Config
from itertools import product as product
from math import sqrt as sqrt
import torch
def default_prior_box():
    mean_layer = []
    for k,f in enumerate(Config.feature_map):
        mean = []
        for i,j in product(range(f),repeat=2):
            f_k = Config.image_size/Config.steps[k]
            cx = (j+0.5)/f_k
            cy = (i+0.5)/f_k

            s_k = Config.sk[k]/Config.image_size
            mean += [cx,cy,s_k,s_k]

            s_k_prime = sqrt(s_k * Config.sk[k+1]/Config.image_size)
            mean += [cx,cy,s_k_prime,s_k_prime]
            for ar in Config.aspect_ratios[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        if Config.use_cuda:
            mean = torch.Tensor(mean).cuda().view(Config.feature_map[k], Config.feature_map[k], -1).contiguous()
        else:
            mean = torch.Tensor(mean).view( Config.feature_map[k],Config.feature_map[k],-1).contiguous()
        mean.clamp_(max=1, min=0)
        mean_layer.append(mean)

    return mean_layer
def encode(match_boxes,prior_box,variances):
    g_cxcy = (match_boxes[:, :2] + match_boxes[:, 2:])/2 - prior_box[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * prior_box[:, 2:])
    # match wh / prior wh
    g_wh = (match_boxes[:, 2:] - match_boxes[:, :2]) / prior_box[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def change_prior_box(box):
    if Config.use_cuda:
        return torch.cat((box[:, :2] - box[:, 2:]/2,     # xmin, ymin
                         box[:, :2] + box[:, 2:]/2), 1).cuda()  # xmax, ymax
    else:
        return torch.cat((box[:, :2] - box[:, 2:]/2,     # xmin, ymin
                         box[:, :2] + box[:, 2:]/2), 1)
# 计算两个box的交集
def insersect(box1,box2):
    label_num = box1.size(0)
    box_num = box2.size(0)
    max_xy = torch.min(
        box1[:,2:].unsqueeze(1).expand(label_num,box_num,2),
        box2[:,2:].unsqueeze(0).expand(label_num,box_num,2)
    )
    min_xy = torch.max(
        box1[:,:2].unsqueeze(1).expand(label_num,box_num,2),
        box2[:,:2].unsqueeze(0).expand(label_num,box_num,2)
    )
    inter = torch.clamp((max_xy-min_xy),min=0)
    return inter[:,:,0]*inter[:,:,1]

def jaccard(box_a, box_b):
    """计算jaccard比
    公式:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    inter = insersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
def point_form(boxes):

    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    """计算default box和实际位置的jaccard比，计算出每个box的最大jaccard比的种类和每个种类的最大jaccard比的box
    Args:
        threshold: (float) jaccard比的阈值.
        truths: (tensor) 实际位置.
        priors: (tensor) default box
        labels: (tensor) 一个图片实际包含的类别数.
        loc_t: (tensor) 需要存储每个box不同类别中的最大jaccard比.
        conf_t: (tensor) 存储每个box的最大jaccard比的类别.
        idx: (int) 当前的批次
    """
    # 计算jaccard比
    overlaps = jaccard(
        truths,
        # 转换priors，转换为x_min,y_min,x_max和y_max
        point_form(priors)
    )
    # [1,num_objects] best prior for each ground truth
    # 实际包含的类别对应box中jaccarb最大的box和对应的索引值，即每个类别最优box
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    # 每一个box,在实际类别中最大的jaccard比的类别，即每个box最优类别
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # 将每个类别中的最大box设置为2，确保不影响后边操作
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # 计算每一个box的最优类别，和每个类别的最优loc
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    # 实现loc的转换，具体的转换公式参照论文中的loc的loss函数的计算公式
    loc = encode(matches, priors,(0.1,0.2))
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    result = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep,0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
if __name__ == '__main__':
    mean = default_prior_box()
    print(mean)