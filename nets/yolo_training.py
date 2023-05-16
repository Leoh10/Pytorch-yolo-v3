import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

class YOLOLoss(nn.Module): 
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.giou           = True
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05  #损失的权重
        self.obj_ratio      = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio      = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda           = cuda

    def clip_by_tensor(self, t, t_min, t_max):#函数的作用是：使t的值变到t_min和t_max之间
        t = t.float()#  
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):#均方误差（Mean square error）---预测值与目标值之间差值平方和的均值
        return torch.pow(pred - target, 2)
    #   torch.pow(x,2)-----返回x的平方值，即求得两个张量的差值平方

    def BCELoss(self, pred, target):#   一般计算pred要经过sigmoid或softmax，转为0,1之间
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)#使得预测值在1e-7~1-1e-7之间
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        #   2分类交叉熵损失，公式如上output所示w表示权重值为1
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   求真实框和预测框所有的iou
        #----------------------------------------------------#
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
        #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        #   计算对角线距离---------计算包裹的面积
        #----------------------------------------------------#
        enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou            = iou - (enclose_area - union_area) / enclose_area
        
        return giou
        
    def forward(self, l, input, targets=None):#该方法有3个输入
        #----------------------------------------------------#
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input代表当前特征层的输出数据的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。ground truth
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #   batchsize 图片数量
        #   13，13有效特征层
        #--------------------------------#
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长stride_h，
        #   每一个特征点对应原来的图片上多少个像素点
        #   416/13=32
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #   stride_h和stride_w都是32。
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   将先验框进行缩放，缩放到特征层尺度大小
        #   先验框的大尺度的有3个先验框，宽和高为
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   view操作得到---->bs, 3*(5+num_classes), 13, 13 ---利用permute函数进行维度变换更换为=> 下一行所示
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        #-----------------------------------------------#
        #   先验框的中心位置的调整参数，
        #   x数据维度为bs,3,13,13,1----->1,3,13,13,1----sigmoid将调整参数转为0,1之间的值
        #   y数据维度为bs,3,13,13,1----->1,3,13,13,1
        #   prediction[..., 0]----前面维度都要，只在最后一个维度上取数据
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])   #   Center x
        y = torch.sigmoid(prediction[..., 1])   #   Center y
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #   w数据维度为bs,3,13,13,1----->1,3,13,13,1
        #-----------------------------------------------#
        w = prediction[..., 2]                  #   宽
        h = prediction[..., 3]                  #   高
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #   conf数据维度为bs,3,13,13,1----->1,3,13,13,1
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])
        #   pred_cls数据维度为bs,3,13,13,num_classes----->1,3,13,13,num_classes----VOC数据集的结果即1,3,13,13,20
        #-----------------------------------------------#
        #   获得网络应该有的预测结果
        #   targets：真实框数据  targets类似于[0.6358, 0.5204, 0.0697, 0.0312, 0.0000]这个样子
        #   scaled_anchors：相对于特征层的先验框大小
        #-----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        #---------------------------------------------------------------#
        #   判断网络应该忽略哪些特征点
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适，保留重合度一般的先验框作为负样本，保证正负样本平衡。
        #   noobj_mask：用于选取哪些先验框不包含物体  无目标为1 有目标为0
        #   box_loss_scale：用于获得xywh的比例 大目标loss权重小，小目标loss权重大 让网络更加去关注小目标
        #   y_true: batch_size, 3, 13, 13, 5 + num_classes  真实框
        #   # y_true的格式[1,3,13,13,25] 并不是每一特征点上都有真实框的，没有真实框的地方为0
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.type_as(x)
            noobj_mask      = noobj_mask.type_as(x)
            box_loss_scale  = box_loss_scale.type_as(x)
        #--------------------------------------------------------------------------#
        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        #--------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale    
        loss        = 0
        obj_mask    = y_true[..., 4] == 1   # 是否有物体，置信度
        n           = torch.sum(obj_mask)
        if n != 0:
            if self.giou:
                #---------------------------------------------------------------#
                #   计算预测结果和真实结果的giou
                #   正样本，编码后的长宽与xy轴偏移量与预测值的差距;使用到了x,y,w,h
                #----------------------------------------------------------------#
                giou        = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                # giou torch.Size([1, 3, 13, 13])------
                #   此时，GIOU作为loss函数时，为L=1−GIOU，当A、B两框不相交时A∪B值不变，最大化GIOU就是就小化C，这样就会促使两个框不断靠近。
                loss_loc    = torch.mean((1 - giou)[obj_mask])
            else:
                #-----------------------------------------------------------#
                #   计算中心偏移情况的loss，使用BCELoss效果好一些，因为计算中心偏移使用了sigmoid,所以不能使用均方差损失，
                #   选用交叉熵损失----最后都求均值
                #-----------------------------------------------------------#
                loss_x      = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y      = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                #-----------------------------------------------------------#
                #   计算宽高调整值的loss
                #-----------------------------------------------------------#
                loss_w      = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h      = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc    = (loss_x + loss_y + loss_h + loss_w) * 0.1
                #   实际存在的框，种类预测结果与实际结果的对比。            
            loss_cls    = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss        += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
                #   置信度损失
        loss_conf   = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss        += loss_conf * self.balance[l] * self.obj_ratio#所有损失相加
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   (gt_box, anchor_shapes)---（真实框，先验框）---输入真实框（0,0，宽，高）
        #   计算真实框的左上角和右下角----
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        #   torch.clamp(input, min, max, out=None) 使得tensor中比min小的变成min，比max大的变成max
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]---扩展为维度和inter维度相同
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]重合部分/（面积相加-重合部分）
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        #   获取真实框及信息的函数
        #   targets=None，l为输入图片数量，为1，anchors为特征图对应的3个先验框在特征图尺度上的大小处理的结果
        #   in_h,in_w表示特征图大小的维度，第一个特征图为13*13
        #-----------------------------------------------------#
        #   计算一共有多少张图片，为1
        #-----------------------------------------------------#
        bs              = len(targets)
        #-----------------------------------------------------#
        #   用于选取哪些特征点上3个先验框是不包含物体的，
        #  noobj_mask =  (bs,3,13,13)
        #   默认先置为全1矩阵，认为所有先验框内部都不包含物体，后续再判断哪些框是包含物体的
        #   requires_grad = False表示这个tensor不参与求导
        #-----------------------------------------------------#
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   参数是进行一个小目标和大目标损失权重的区分，将小目标框权重进行增强，大目标框权重进行减弱，
        #   让网络更加去关注小目标
        #   box_loss_scale = (bs,3,13,13)
        #-----------------------------------------------------#
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   self.bbox_attrs     = 5 + num_classes
        #   y_true指的是图片的真实框的参数，维度为1,3,13,13，5+num_classes---0矩阵
        #   y_true = 1,3,13,13，25
        #-----------------------------------------------------#
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):    
        #   若没有输入图片则跳过循环，执行其他操作        
            if len(targets[b])==0:
                continue
            #   利用zeros_like生成相同维度大小的x,初始值为0
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #   由于真实框是归一化的结果，乘特征层的宽高，就可以将真实框映射到特征层上
            #   targets[1][:,[0,2]]表示是真实目标的x1,x2-------------标签数据都是x1,y1,x2,y2--对应位置0,1,2,3
            #   targets[1][:,[1,3]]表示是真实目标的y1,y2
            #   targets：[0.6358, 0.5204, 0.0697, 0.0312, 0.0000] ：x,y,h,w,置信度
            #-------------------------------------------------------#
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]#---数据框的种类
            batch_target = batch_target.cpu()#把数据转移到cpu上
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式----这里转换是为了数据可以送到这个函数 self.calculate_iou  计算重合度
            #   num_true_box, 4
            #   torch.FloatTensor([1,2])，可以将变量转为浮点型32位，转换的变量类型为列表或数组
            #   torch.cat((x,y),1),可以将x(1,2),y[:,2:4]表示2,3位置数据----gt_box==[0,0,batch_target[:, 2:4]],4个维度，只要宽高值，便于计算IOU
            #   按维数0拼接（竖着拼） C = torch.cat( (A,B),0 )
            #   按维数1拼接（横着拼） C = torch.cat( (A,B),1 )
            #   batch_target[:, 2:4]----start:end:step,范围前闭后开
            #-------------------------------------------------------#
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            #   torch.zeros((len(anchors), 2))-----zeros(9,2)-----
            #   缩放后的9个先验框的宽高：anchors = [(0.3125, 0.40625), (0.5, 0.9375), (1.03125, 0.71875), 
            #   (0.9375, 1.90625), (1.9375, 1.40625), (1.84375, 3.71875), 
            #   (3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
            #   长度为9，利用zeros和格式转换得到anchor_shapes大小为9,4---数据为
            #  anchor_shapes =  tensor([[ 0.0000,  0.0000,  0.3125,  0.4062],
            #    [ 0.0000,  0.0000,  0.5000,  0.9375],
            #    [ 0.0000,  0.0000,  1.0312,  0.7188],
            #   [ 0.0000,  0.0000,  0.9375,  1.9062],
            #   [ 0.0000,  0.0000,  1.9375,  1.4062],
            #   [ 0.0000,  0.0000,  1.8438,  3.7188],
            #   [ 0.0000,  0.0000,  3.6250,  2.8125],
            #   [ 0.0000,  0.0000,  4.8750,  6.1875],
            #   [ 0.0000,  0.0000, 11.6562, 10.1875]])
            #-------------------------------------------------------#
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 3]每一个真实框和对应特征尺度上的3个先验框的重合情况
            #   best_ns:返回两个值
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            #   argmax(x,dim=-1),返回指定维度的最大值的序号

            #-------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            #   下面的循环每次会取两个值，t是每个真实框最大的重合度max_iou,best_n指的是每个真实框最重合的先验框的序号
            #   enumerate（sequence,[start=0]）-----将一个可遍历的数据对象（列表，元组或字符串）组合为一个索引序列，同时列出数据和下标
            for t, best_n in enumerate(best_ns):
            #   判断这个先验框的序号是否属于当前这个特征层，
                if best_n not in self.anchors_mask[l]:
                    continue
                #   anchors_mask[l]=[6,7,8]
                #   anchors_mask[l]=[3,4,5]
                #   anchors_mask[l]=[0,1,2]
                #----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框------共有3个先验框，真实框属于是哪个尺度下第几个先验框
                #----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                #----------------------------------------#
                #   获得真实框属于哪个网格点，计算出真实坐标
                #   long()表示向下取整可以找到左上角的网格点，因为yolo v3是根据左上角的网格点进行预测的
                #   i,j是当前先验框对应特征层的哪一个网格点
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   取出真实框的种类
                #----------------------------------------#
                c = batch_target[t, 4].long()

                #----------------------------------------#
                #   noobj_mask代表无目标的特征点，
                #   在前面定义了他的数据维度都是1，这里就利用目标检测确定的点位来置为0，表示有真实框存在于这个特征点
                #----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                #----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #----------------------------------------#
                if not self.giou:
                    #----------------------------------------#
                    #   tx、ty代表中心调整参数的真实值
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    #   获得h,w方向的应该有的调整参数，
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    #   对4序号
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    #----------------------------------------#
                                        #   tx、ty代表中心调整参数的真实值
                    #   b:对应那张图片
                    #   k:哪一个先验框（0-2）
                    #   i:真实框位置
                    #   j:真实框位置
                    #   tx、ty代表中心调整参数的真实值
                    #   y_true = 1,3,13,13，25，则结果就是哪有1，是哪一类
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #   batch_target[t, 2] * batch_target[t, 3] ----宽  * 高
                #   特征层维度大小--13/13-- in_w / in_h
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale
                # 根据网络的预测结果获得预测框，计算预测框和所有真实框的重合程度，如果重合程度大于一定门限，则将该预测框对应的先验框忽略。其余作为负样本。
    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #   先根据预测结果进行一个解码
        #-----------------------------------------------------#
        bs = len(targets)   #   共一张图片

        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角-------
        #   torch.linspace 函数的作用是，返回一个一维的tensor（张量）[start开始值 ,end结束值,steps：分割的点数，默认是100]
        #   linspace(0,13-1,13)即生成0-12个共13个数据，重复行的维数为in_h，即生成in_w*in_h大小的二维网格，间距为1
        #   以特征层13*13为例  grid_x为 【0-12】
        #   grid_x torch.Size([1, 3, 13, 13])
        #   grid_y torch.Size([1, 3, 13, 13])
        #   view(x.shape)将数据调整为与x维度大小一致，数据类型一致type_as(x)
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        #   生成先验框的宽高
        #   anchor_h torch.Size([1, 3, 13, 13])
        #   anchor_w torch.Size([1, 3, 26, 26])
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]] # 从9个先验框里选出相应特征层的3个先验框
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #   现在anchor_w，anchor_h  格式如上，里面的值是相应先验框的大小
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        #   若dim为负，则将被转化为dim+input.dim()+1，即2，即倒数第一个维度
        #   对batchsize进行一个循环-----对一张图片进行循环
        #   ---------------------------------------------------------------------------------------------------
        for b in range(bs):           
            #-------------------------------------------------------#
            #   若真实框与先验框计算重合度，若计算重合度大于某一个设定阈值，
            #   则将先验框设定为打算忽略的先验框
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                #   torch.max(x,dim=0)，2维数据，0表示行间对比，dim=1表示列间比较
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                #   下面这行代码，即设置重合度大的先验框为忽略的先验框，而将剩余先验框设置为负样本，保证正负样本的平衡
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0#设置0表示有目标，1表示无目标
        return noobj_mask, pred_boxes

def weights_init(net, init_type='normal', init_gain = 0.02):#假如不利用预训练权重和主干权重，
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            #   if语句头的意思是，m中是否有weight这个属性并且m的类名中是否有"Conv"。
            if init_type == 'normal':
                # 从给定均值和标准差的正态分布N(0.0, init_gain)中生成值，填充输入的m.weight.data
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                # 用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，
                # 标准差为gain * sqrt(2/(fan_in + fan_out))的正态分布。也被称为Glorot initialisation.
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # 用一个正态分布生成值，填充输入的张量或变量。结果张量中的值采样自均值为0，
                # 标准差为sqrt(2/((1 + a^2) * fan_in))的正态分布。
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                # 用（半）正交矩阵填充输入的张量或变量。输入张量必须至少是2维的，对于更高维度的张量，
                # 超出的维度会被展平，视作行等于第一个维度，列等于稀疏矩阵乘积的2维表示。其中非零元素生成自均值为0，
                # 标准差为std的正态分布。
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            #   elif语句头的意思是，m的；类名中是否有"BatchNorm2d"。
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            # 用val的值(0.0)填充输入的张量或变量(m.bias.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
