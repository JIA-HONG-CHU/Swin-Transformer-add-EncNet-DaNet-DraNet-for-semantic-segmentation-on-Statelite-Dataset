import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout,PairwiseDistance 
from torch.nn import functional as F
from torch.autograd import Variable

import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale

from mmseg.core import add_prefix
from ..builder import HEADS
from .decode_head import BaseDecodeHead

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class CPAMEnc(Module):
    """
    CPAM encoding module
    """
    def __init__(self, in_channels, norm_layer, conv_cfg , act_cfg):
        super(CPAMEnc, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)
        
        self.conv1 = ConvModule(in_channels, in_channels, 1, bias=False, conv_cfg = conv_cfg, norm_cfg= norm_layer, 
                                act_cfg = act_cfg)
        self.conv2 = ConvModule(in_channels, in_channels, 1, bias=False, conv_cfg = conv_cfg, norm_cfg= norm_layer, 
                                act_cfg = act_cfg)
        self.conv3 = ConvModule(in_channels, in_channels, 1, bias=False, conv_cfg = conv_cfg, norm_cfg= norm_layer, 
                                act_cfg = act_cfg)
        self.conv4 = ConvModule(in_channels, in_channels, 1, bias=False, conv_cfg = conv_cfg, norm_cfg= norm_layer, 
                                act_cfg = act_cfg)
#         self.conv1 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
#                                 norm_layer(in_channels),
#                                 ReLU(True))
#         self.conv2 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
#                                 norm_layer(in_channels),
#                                 ReLU(True))
#         self.conv3 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
#                                 norm_layer(in_channels),
#                                 ReLU(True))
#         self.conv4 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
#                                 norm_layer(in_channels),
#                                 ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4), 2)


class CPAMDec(Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

        self.conv_query = Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key = Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value = Linear(in_channels, in_channels) # value2
    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize,C,width ,height = x.size()
        m_batchsize,K,M = y.size()

        proj_query  = self.conv_query(x).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key =  self.conv_key(y).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        energy =  torch.bmm(proj_query,proj_key)#BxNxK
        attention = self.softmax(energy) #BxNxk

        proj_value = self.conv_value(y).permute(0,2,1) #BxCxK
        out = torch.bmm(proj_value,attention.permute(0,2,1))#BxCxN
        out = out.view(m_batchsize,C,width,height)
        out = self.scale*out + x
        return out


class CCAMDec(Module):
    """
    CCAM decoding module
    """
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape #BXC1XN
        proj_key  = y_reshape.permute(0,2,1) #BX(N)XC
        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) #BCN
        
        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out


class CLGD(Module):
    """
    Cross-level Gating Decoder
    """
    def __init__(self, in_channels, out_channels, norm_layer, conv_cfg, act_cfg):
        super(CLGD, self).__init__()

        inter_channels= 32
        back_channels = 256
        
        self.conv_low = ConvModule(in_channels, inter_channels, 3, padding=1, bias=False, conv_cfg = conv_cfg,
                                   norm_cfg= norm_layer, act_cfg = act_cfg)
        
        self.conv_cat = ConvModule(back_channels+inter_channels, back_channels, 3, padding=1,
                                   bias=False, conv_cfg = conv_cfg, norm_cfg= norm_layer, act_cfg = act_cfg)
        
        self.conv_att = ConvModule(back_channels+inter_channels , 1, 1,
                                    conv_cfg = conv_cfg, act_cfg = dict(type='Sigmoid'))
        
        self.conv_out = ConvModule(back_channels, out_channels, 3, padding=1, bias=False, norm_cfg= norm_layer,
                                    conv_cfg = conv_cfg, act_cfg = act_cfg)
        
#         self.conv_low = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU()) #skipconv
        
#         self.conv_cat = nn.Sequential(nn.Conv2d(in_channels+inter_channels, in_channels, 3, padding=1, bias=False),
#                                      norm_layer(in_channels),
#                                      nn.ReLU()) # fusion1

#         self.conv_att = nn.Sequential(nn.Conv2d(in_channels+inter_channels, 1, 1),
#                                     nn.Sigmoid()) # att

#         self.conv_out = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
#                                      norm_layer(out_channels),
#                                      nn.ReLU()) # fusion2
        self._up_kwargs = up_kwargs

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x,y):
        """
            inputs :
                x : low level feature(N,C,H,W)  y:high level feature(N,C,H,W)
            returns :
                out :  cross-level gating decoder feature
        """
        low_lvl_feat = self.conv_low(x)
        high_lvl_feat = upsample(y, low_lvl_feat.size()[2:], **self._up_kwargs)
        feat_cat = torch.cat([low_lvl_feat,high_lvl_feat],1)
      
        low_lvl_feat_refine = self.gamma*self.conv_att(feat_cat)*low_lvl_feat 
        low_high_feat = torch.cat([low_lvl_feat_refine,high_lvl_feat],1)
        low_high_feat = self.conv_cat(low_high_feat)
        
        low_high_feat = self.conv_out(low_high_feat)

        return low_high_feat

    
    
@HEADS.register_module()
class DRANHead(BaseDecodeHead):
   

    def __init__(self, **kwargs):
        super(DRANHead, self).__init__(input_transform='multiple_select',**kwargs)
        
         ## Convs or modules for CPAM 
        self.conv_cpam_b = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.cpam_enc = CPAMEnc(self.channels,norm_layer=self.norm_cfg, conv_cfg=self.conv_cfg,act_cfg=self.act_cfg)
        self.cpam_dec = CPAMDec(self.channels) 
        
        self.conv_cpam_e = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        ## Convs or modules for CCAM
        self.conv_ccam_b = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.ccam_enc = ConvModule(
            self.channels,
            self.channels//16,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.ccam_dec = CCAMDec() 
        
        self.conv_ccam_e = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        ## Fusion conv
        self.conv_cat = ConvModule(
            self.channels*2,
            self.channels//2,
            3,
            padding=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        
         ## Cross-level Gating Decoder(CLGD)  #swin-small\swin-tiny 96 swin-base 128 swin-large 198
        self.clgd = CLGD(128,self.channels//2,norm_layer=self.norm_cfg ,conv_cfg=self.conv_cfg,act_cfg=self.act_cfg)
        
        #cls_seg
        self.conv_seg = nn.Conv2d(self.channels//2, self.num_classes, kernel_size=1)
        
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
        
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        x = x[-1] #selcet last one feature
        
        ## Compact Channel Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(x)
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)
        
        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)  
        
        ## Fuse two modules
        ccam_feat = self.conv_ccam_e(ccam_feat)
        cpam_feat = self.conv_cpam_e(cpam_feat)
        
        feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))
        
        ## Cross-level Gating Decoder(CLGD)
        final_feat = self.clgd(inputs[0], feat_sum) 
        final_feat = self.cls_seg(final_feat)
        
        return final_feat
    
     