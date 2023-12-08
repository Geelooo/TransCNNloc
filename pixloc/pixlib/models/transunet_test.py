"""
    搭建Unet3+++网络
"""
import torchvision
import torch
import torch.nn as nn

from pixloc.pixlib.models.base_model import BaseModel
from pixloc.pixlib.models.utils import checkpointed
from swin_hybrid_test import BasicLayer

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from pixloc.pixlib.models.lys_layers import unetConv2
from pixloc.pixlib.models.lys_init_weights import init_weights

class DecoderBlock(nn.Module):
    def __init__(self, previous, skip, out, num_convs=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous+skip if i == 0 else out, out,
                kernel_size=3, padding=1, bias=norm is None)
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        upsampled = self.upsample(previous)
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        assert (hu <= hs) and (wu <= ws), 'Using ceil_mode=True in pooling?'
        # assert (hu == hs) and (wu == ws), 'Careful about padding'
        skip = skip[:, :, :hu, :wu]
        return self.layers(torch.cat([upsampled, skip], dim=1))



class TransUnet(nn.Module):

    default_conf = {
        'output_scales': [0, 2, 4],  # what scales to adapt and output
        'output_dim': 128,  # # of channels in output feature maps
        'encoder': 'vgg19',  # string (torchvision net) or list of channels
        'num_downsample': 4,  # how many downsample block (if VGG-style net)
        'decoder': [64, 64, 64, 64],  # list of channels of decoder
        'decoder_norm': 'nn.BatchNorm2d',  # normalization ind decoder blocks
        'do_average_pooling': False,
        'compute_uncertainty': False,
        'checkpointed': False,  # whether to use gradient checkpointing
    }

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.n_channels = 3                 # 通道数
        self.bilinear = 'bilinear'          # 双线性插值
        self.feature_scale = 4              # 特征层数        
        self.is_batchnorm = False           # 输出归一化
        self.output_dim=[0,2,4]
        # self.scales = [2**s for s in conf.output_scales]
        self.filters = [64, 128, 256, 512, 512]
        self.flatten_dim=384
        skip_dims=[64,128,256,512,512]
        conf_decoder=[64, 64, 64, 32]

        ## --------------------------Decoder--------------------------
        Block = checkpointed(DecoderBlock, do=True)
        norm = eval('nn.BatchNorm2d')

        previous = skip_dims[-1]
        decoder = []
        for out, skip in zip(conf_decoder, skip_dims[:-1][::-1]):
            decoder.append(Block(previous, skip, out, norm=norm))
            previous = out
        self.decoder = nn.ModuleList(decoder)

        # feature
        self.outconv1 = nn.Conv2d(32, 32, 1)
        self.outconv2 = nn.Conv2d(64, 128, 1)
        self.outconv3 = nn.Conv2d(512, 128, 1)

        # uncertainty
        self.uncer1=nn.Conv2d(32,1,1,1)
        self.uncer2=nn.Conv2d(64,1,1,1)
        self.uncer3=nn.Conv2d(512,1,1,1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        
        ## --------------------------Encoder---------------------------
        model1=torchvision.models.vgg19(pretrained=True)
        # level-1
        self.block1 = nn.Sequential(model1.features[0],model1.features[1],model1.features[2],model1.features[3])
        # level-2
        self.block2 = nn.Sequential(model1.features[4],model1.features[5],model1.features[6],model1.features[7],model1.features[8])
        # level-3
        self.block3 = nn.Sequential(model1.features[9],model1.features[10],model1.features[11],model1.features[12],model1.features[13],model1.features[14],model1.features[15],model1.features[16],model1.features[17])
        # level-4
        self.block4 = nn.Sequential(model1.features[18],model1.features[19],model1.features[20],model1.features[21],model1.features[22],model1.features[23],model1.features[24],model1.features[25],model1.features[26])
        # level-5
        self.block5 = nn.Sequential(model1.features[27],model1.features[28],model1.features[29],model1.features[30],model1.features[31],model1.features[32],model1.features[33],model1.features[34],model1.features[35])

        # Swin transblock
        self.flatten1=nn.Conv2d(self.filters[-1], self.flatten_dim, 1, 1)
        self.swin_block = BasicLayer(dim=96*4,
                                     depth=6,
                                     num_heads=12,
                                     window_size=7,
                                     mlp_ratio=4.,
                                     qkv_bias=True,
                                     drop=0.,
                                     attn_drop=0.,
                                     drop_path=[0.036363635212183, 0.045454543083906174, 0.054545458406209946, 0.06363636255264282, 0.0727272778749466, 0.08181818574666977],
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        self.flatten2=nn.Conv2d(self.flatten_dim, self.filters[-1], 1, 1)

        pretrained_path='/home/lys/Workplace/python/TransUNet/datasets/upernet_swin_tiny_patch4_window7_512x512.pth'
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.swin_block.state_dict()
        for key,value in pretrained_dict.items():
            if "backbone.layers.2.blocks" in key:
                tem_key=key[18:]
                model_dict.update({tem_key:value})
        self.swin_block.load_state_dict(model_dict,strict=True)
                    
    # def _forward(self, data):
    def forward(self, inputs):
        ## --------------------------Encoder--------------------------
        # image = data['image']
        image=inputs
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        # inputs 3*384*512

        skip_features = []

        h1 = self.block1(image)  # h1->64*384*512
        skip_features.append(h1)

        h2 = self.block2(h1)  # h2->128*192*256
        skip_features.append(h2)

        h3 = self.block3(h2)  # h3->256*96*128
        skip_features.append(h3)

        h4 = self.block4(h3)  # h4->512*48*64
        skip_features.append(h4)

        hd5 = self.block5(h4)  # h5->512*24*32      --->

        hd5=self.flatten1(hd5) # h5->384*32*32    线性变换
        B,C,H,W=hd5.shape
        hd5 = hd5.flatten(2)  # h5->1*384*1024
        hd5 = hd5.transpose(-1, -2)  # (B, n_patches, hidden
        hd5,H,W=self.swin_block(hd5,H,W)    # h5->1*1024*384
        hd5=hd5.view(B, H, W, -1).permute(0,3,1,2)  # h5->1*384*32*32
        hd5=self.flatten2(hd5)  #h5->1*512*32*32    <---
        skip_features.append(hd5)
        ## --------------------------Decoder--------------------------
        pre_features = [skip_features[-1]]
        for block, skip in zip(self.decoder, skip_features[:-1][::-1]):# 这里会用到decoder的forward
            pre_features.append(block(pre_features[-1], skip))
        pre_features = pre_features[::-1]  # fine to coarse

        feature_map_1 = self.outconv1(pre_features[0])  # feature1->32*384*512
        feature_map_3 = self.outconv2(pre_features[2])  # feature3->128*96*128
        feature_map_5 = self.outconv3(pre_features[4])  # feature5->128*24*32

        uncertainty_map1=self.uncer1(pre_features[0])   # un1->1*384*512
        uncertainty_map3=self.uncer2(pre_features[2])   # un3->1*96*128
        uncertainty_map5=self.uncer3(pre_features[4])   # un5->1*24*32

        out_features = []
        out_features.append(feature_map_1)
        out_features.append(feature_map_3)
        out_features.append(feature_map_5)
        pred = {'feature_maps': out_features}

        # if self.conf.compute_uncertainty:
        #     confidences = []
        #     conf = torch.sigmoid(uncertainty_map1)
        #     confidences.append(conf)
        #     conf = torch.sigmoid(uncertainty_map3)
        #     confidences.append(conf)
        #     conf = torch.sigmoid(uncertainty_map5)
        #     confidences.append(conf)
        # pred['confidences'] = confidences
        # return pred
        return pre_features[0],pre_features[2],pre_features[4], uncertainty_map1, uncertainty_map3, uncertainty_map5


    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
    input = torch.randn((1, 3, 512, 512))
    model = TransUnet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # input=input.cuda(device)
    # model.to(device)
    f1,f3,f5,u1,u3,u5 = model(input)
    # print(f1,f3,f5,u1,u3,u5)
    print(f1.shape,f3.shape,f5.shape,"\n",u1.shape,u3.shape,u5.shape)