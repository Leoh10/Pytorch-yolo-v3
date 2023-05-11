from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    #以第7层数据为例，构建特征金字塔，先进行一个1*1卷积，降低通道数，再进行3*3卷积扩展通道数）
    #输入为 last_layer0= make_last_layers([512,1024],1024,75)
        # conv2d(1024, 512, 1),
        # conv2d(512, 1024, 3),
        # conv2d(1024, 512, 1),
        # conv2d(512, 1024, 3),
        # conv2d(1024, 512, 1),
        #上面为5次conv 2d卷积块
        # conv2d(512, 1024, 3)3----一次3*3卷积块
        # nn.Conv2d(1024, 75, kernel_size=1, stride=1, padding=0, bias=True)#--1*1卷积进行分类预测，与回归预测
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        # #实例化darknet53(),返回out3,out4,out5
        #---------------------------------------------------#
        self.backbone = darknet53()

        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #   out_filters=[64, 128, 256, 512, 1024]，-1，-2，-3，即倒着取list数据，依次为1024,512,256
        #   anchors_mask从代码及结果看值设定为[3,3,3],存在3个先验框，voc数据集目标类别是20类;
        #  last_layer0= make_last_layers([512,1024],1024,75)
        #  last_layer1= make_last_layers([256, 768],512,75)
        #   last_layer2=make_last_layers([128, 384],256,75)
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)#1*1卷积调整通道数
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')#利用上采样将特征图扩张为26*26
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；---第五层
        #   26,26,512；---第六层
        #   13,13,1024；--第七层
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # 单独把5次卷积的操作保存在out0_branch中
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #   torch.cat(tensors,dim,out=None)；将张量按给定的维度连接，dim=0,表示将张量x,y按列连起来
        #   dim=1,表示将张量x,y按行连接起来；
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        #   out0---13,13,75-----随后利用三个矩阵来判断先验框是否包含物体以及对物体进行分类预测与先验框的调整参数的回归
        #   out1---26,26,75
        #   out2---52,52,75
        return out0, out1, out2