import sys
#!pip install './pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/'
sys.path.append('./pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/')

#!pip install './efficientnet_pytorch-0.7.1/efficientnet_pytorch-0.7.1/'
sys.path.append('./efficientnet_pytorch-0.7.1/efficientnet_pytorch-0.7.1/')

import torch
from torch import nn, optim
import timm
import segmentation_models_pytorch as smp
from pdb import set_trace as st
class SegmentationModel(nn.Module):
    def __init__(self, backbone=None, segtype='unet', pretrained=False):
        super(SegmentationModel, self).__init__()
        
        n_blocks = 4
        self.n_blocks = n_blocks
        
        self.encoder = timm.create_model(
            'resnet18d',
            in_chans=3,
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        
        x = torch.stack([x]*3, 1)
        
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
    
#from timm.models.layers.conv2d_same import Conv2dSame
class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return timm.models.layers.conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def conv3d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0), dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """ Tensorflow like 'SAME' convolution wrapper for 3d convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv3d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv3d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv3dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv3d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output

class RSNA_2023_1st_Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super(RSNA_2023_1st_Model, self).__init__()
        
        drop = 0.2
        
        self.encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)

        feats = self.encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        lstm_embed = 512
        
        self.lstm = nn.LSTM(feats, lstm_embed, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_embed*2, lstm_embed),
            nn.BatchNorm1d(lstm_embed),
            nn.Dropout(drop),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, num_classes),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        feat = self.encoder(x)
        
        feat = self.avgpool(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1)
        
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        feat, _ = torch.max(feat, dim=1)
        feat = torch.nan_to_num(feat, 0, 0)
        # st()
        
        return feat
    
class RSNA_2023_1st_Model2(nn.Module):
    def __init__(self, pretrained=True, mask_head=False, num_classes=3):
        super(RSNA_2023_1st_Model2, self).__init__()
        
        self.mask_head = mask_head
        
        drop = 0.2
        
        true_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)

        segmentor = smp.Unet(f"tu-tf_efficientnetv2_s", encoder_weights=None, in_channels=3, classes=3)
        self.encoder = segmentor.encoder
        self.decoder = segmentor.decoder
        self.segmentation_head = segmentor.segmentation_head
        
        st = true_encoder.state_dict()

        self.encoder.model.load_state_dict(st, strict=False)
        
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        
        feats = true_encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            nn.Linear(lstm_embed, lstm_embed//2),
            nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(drop),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed//2, num_classes),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(x)
        
        if self.mask_head:
        
            decoded = self.decoder(*features)
        
            masks = self.segmentation_head(decoded)
        
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        feat, _ = torch.max(feat, dim=1)

        if self.mask_head:
            return feat, masks
        else:
            return feat
        
class RSNA_2023_1st_Model3(nn.Module):
    def __init__(self, pretrained=False, mask_head=False, n_classes=9):
        super(RSNA_2023_1st_Model3, self).__init__()
        
        self.mask_head = mask_head
        
        drop = 0.
        
        true_encoder = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)

        segmentor = smp.Unet(f"tu-tf_efficientnetv2_s.in21k_ft_in1k", encoder_weights=None, in_channels=3, classes=3)
        self.encoder = segmentor.encoder
        self.decoder = segmentor.decoder
        self.segmentation_head = segmentor.segmentation_head
        
        st = true_encoder.state_dict()

        self.encoder.model.load_state_dict(st, strict=False)
        
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        
        feats = true_encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, n_classes),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(x)
        
        if self.mask_head:
        
            decoded = self.decoder(*features)
        
            masks = self.segmentation_head(decoded)
        
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks
        else:
            return feat

        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('/kaggle/input/contrails-model-def1/src_inference1')
# from collections import OrderedDict
# from coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
# from layers import *


class RSNA_2023_1st_Model4(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=10, seg_classes=4, ps=0,mask_head=True, **kwargs):
        super().__init__()
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 

        feats = 512
        drop = 0.0
        self.mask_head = mask_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, num_classes),
        )
        
#         if pre is not None:
#             sd = torch.load(pre)['model']
#             print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, seg_classes, blur=True))
        self.up_result=2
    
    def forward(self, x):
        # x = x[:,:,:5].contiguous()
        # nt = x.shape[2]
        # x = x.permute(0,2,1,3,4).flatten(0,1)
        # x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        # print(encs[3].view(-1,*encs[3].shape[1:]).shape)
        # encs = [encs[0].view(-1,*encs[0].shape[1:])[:,-1], 
        #         encs[1].view(-1,*encs[1].shape[1:])[:,-1], 
        #         self.lstm[-2](encs[2].view(-1,*encs[2].shape[1:]))[:,-1],
        #         self.lstm[-1](encs[3].view(-1,*encs[3].shape[1:]))[:,-1]]
        dec4 = encs[-1]
        if self.mask_head:
        
            dec3 = self.dec4(dec4,encs[-2])
            dec2 = self.dec3(dec3,encs[-3])
            dec1 = self.dec2(dec2,encs[-4])

            # print(dec4.shape)
            x = self.fpn([dec4, dec3, dec2], dec1)
            x = self.final_conv(self.drop(x))
            if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        feat = dec4
        avg_feat = self.avgpool(feat)
        # print(avg_feat.shape)
        
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm2(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, x
        else:
            return feat
        
        # return feat, x
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
# from src.coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
# from src.layers import *

class RSNA_2023_1st_Model5(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=4, ps=0, mask_head=False, **kwargs):
        super().__init__()
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 

        feats = 512
        drop = 0.0
        self.mask_head = mask_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed*2, 10),
        )
        
#         if pre is not None:
#             sd = torch.load(pre)['model']
#             print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)

        self.mask_head_3 = self.get_mask_head(nc[-2])
        self.mask_head_4 = self.get_mask_head(nc[-1])

        
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, 4, blur=True))
        self.up_result=2

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
    
    def forward(self, x):
        # x = x[:,:,:5].contiguous()
        # nt = x.shape[2]
        # x = x.permute(0,2,1,3,4).flatten(0,1)
        # x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        # print(encs[3].view(-1,*encs[3].shape[1:]).shape)
        # encs = [encs[0].view(-1,*encs[0].shape[1:])[:,-1], 
        #         encs[1].view(-1,*encs[1].shape[1:])[:,-1], 
        #         self.lstm[-2](encs[2].view(-1,*encs[2].shape[1:]))[:,-1],
        #         self.lstm[-1](encs[3].view(-1,*encs[3].shape[1:]))[:,-1]]
        dec4 = encs[-1]

        # print(encs[-1].shape)
        # print(encs[-2].shape)
        

        if self.mask_head:
            masks1 = self.mask_head_4(encs[-1])
            masks1 = F.interpolate(masks1,size=CFG.image_size,mode='bilinear')
            
            masks2 = self.mask_head_3(encs[-2])
            masks2 = F.interpolate(masks2,size=CFG.image_size,mode='bilinear')

            # print(masks1.shape)
            # print(masks2.shape)
            
            # dec3 = self.dec4(dec4,encs[-2])
            # dec2 = self.dec3(dec3,encs[-3])
            # dec1 = self.dec2(dec2,encs[-4])
    
            # # print(dec4.shape)
            # x = self.fpn([dec4, dec3, dec2], dec1)
            # x = self.final_conv(self.drop(x))
            # if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        
        feat = dec4
        avg_feat = self.avgpool(feat)
        # print(avg_feat.shape)
        
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm2(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks1, masks2
        else:
            return feat

        
class RSNA_2023_1st_Model6(nn.Module):
    def __init__(self, pretrained=True, mask_head=False, n_classes=2):
        super(Model6, self).__init__()
        
        self.mask_head = mask_head
        
        drop = 0.
        
        true_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, in_chans=3,
                                         global_pool='', num_classes=0,
                                         drop_rate=drop, drop_path_rate=drop)

        segmentor = smp.Unet(f"tu-tf_efficientnetv2_s.in21k_ft_in1k", encoder_weights=None, in_channels=3, classes=4)
        self.encoder = segmentor.encoder
        # self.decoder = segmentor.decoder
        # self.segmentation_head = segmentor.segmentation_head
        
        st = true_encoder.state_dict()

        self.encoder.model.load_state_dict(st, strict=False)
        
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        
        feats = true_encoder.num_features
        # feats = 256
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)


        self.mask_head_3 = self.get_mask_head(true_encoder.feature_info[-2]['num_chs'])
        self.mask_head_4 = self.get_mask_head(true_encoder.feature_info[-1]['num_chs'])

        print('total channels in seg head: ',true_encoder.feature_info[-2]['num_chs'],true_encoder.feature_info[-1]['num_chs'] )
        
        lstm_embed = feats * 1
        
        self.lstm = nn.GRU(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, n_classes),
        )

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
    
    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(x)
        # print(i.shape for i in features)
        # print(features[-1].shape)
        # print(features[-2].shape)
        
        if self.mask_head:
            masks1 = self.mask_head_4(features[-1])
            masks1 = nn.functional.interpolate(masks1,size=CFG.image_size,mode='bilinear')
            
            masks2 = self.mask_head_3(features[-2])
            masks2 = nn.functional.interpolate(masks2,size=CFG.image_size,mode='bilinear')
            
            # decoded = self.decoder(*features)
        # 
            # masks2 = self.segmentation_head(decoded)
        
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        # print(feat.shape)
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks1, masks2
        else:
            return feat        

        
class RSNA_2023_1st_Model7(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=4, ps=0,mask_head=True, **kwargs):
        super().__init__()
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 

        feats = 512
        drop = 0.0
        self.mask_head = mask_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed*2, 10),
        )
        
        if pre is not None:
            sd = torch.load(pre)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)

        self.mask_head_3 = self.get_mask_head(nc[-2])
        self.mask_head_4 = self.get_mask_head(nc[-1])

        
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, num_classes, blur=True))
        self.up_result=2

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
    
    def forward(self, x):
        # x = x[:,:,:5].contiguous()
        # nt = x.shape[2]
        # x = x.permute(0,2,1,3,4).flatten(0,1)
        # x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        # x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        # print(encs[3].view(-1,*encs[3].shape[1:]).shape)
        # encs = [encs[0].view(-1,*encs[0].shape[1:])[:,-1], 
        #         encs[1].view(-1,*encs[1].shape[1:])[:,-1], 
        #         self.lstm[-2](encs[2].view(-1,*encs[2].shape[1:]))[:,-1],
        #         self.lstm[-1](encs[3].view(-1,*encs[3].shape[1:]))[:,-1]]
        dec4 = encs[-1]

        # print(encs[-1].shape)
        # print(encs[-2].shape)
        

        if self.mask_head:
            # masks1 = self.mask_head_4(encs[-1])
            # masks1 = F.interpolate(masks1,size=CFG.image_size,mode='bilinear')
            
            # masks2 = self.mask_head_3(encs[-2])
            # masks2 = F.interpolate(masks2,size=CFG.image_size,mode='bilinear')

            # print(masks1.shape)
            # print(masks2.shape)
            
            dec3 = self.dec4(dec4,encs[-2])
            dec2 = self.dec3(dec3,encs[-3])
            dec1 = self.dec2(dec2,encs[-4])
    
            # print(dec4.shape)
            x = self.fpn([dec4, dec3, dec2], dec1)
            x = self.final_conv(self.drop(x))
            if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        
        feat = dec4
        avg_feat = self.avgpool(feat)
        # print(avg_feat.shape)
        
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm2(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        # feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, x
        else:
            return feat
