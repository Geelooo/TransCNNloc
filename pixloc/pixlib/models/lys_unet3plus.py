"""
    搭建Unet3+++网络
"""
import torchvision
import torch
import torch.nn as nn

from pixloc.pixlib.models.base_model import BaseModel
from pixloc.pixlib.models.utils import checkpointed


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from pixloc.pixlib.models.lys_layers import unetConv2
from pixloc.pixlib.models.lys_init_weights import init_weights


class UNet3Plus(BaseModel):
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
    def _init(self, conf):
        self.n_channels = 3                 # 通道数
        self.bilinear = 'bilinear'          # 双线性插值
        self.feature_scale = 4              # 特征层数        
        self.is_batchnorm = False           # 输出归一化
        self.scales = [2**s for s in conf.output_scales]
        filters = [64, 128, 256, 512, 512]

        ## --------------------------Decoder--------------------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->384*512, hd4->48*64
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->192*256, hd4->48*64
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->96*128, hd4->48*64
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->48*64, hd4->48*64
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->24*32, hd4->48*64
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode=self.bilinear)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->384*512, hd3->96*128
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->192*256, hd3->96*128
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->96*128, hd3->96*128
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->48*64, hd4->96*128
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode=self.bilinear) 
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->24*32, hd4->96*128
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode=self.bilinear) 
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->384*512, hd2->192*256
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->192*256, hd2->192*256
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->96*128, hd2->192*256
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode=self.bilinear)
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->48*64, hd2->192*256
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode=self.bilinear) 
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->24*32, hd2->192*256
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode=self.bilinear) 
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->384*512, hd1->384*512
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->192*256, hd1->384*512
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode=self.bilinear)
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->96*128, hd1->384*512
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode=self.bilinear)
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->48*64, hd1->384*512
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode=self.bilinear)
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->24*32, hd1->384*512
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode=self.bilinear)
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # feature
        self.outconv1 = nn.Conv2d(320, 32, 1)
        self.outconv2 = nn.Conv2d(320, 128, 1)
        self.outconv3 = nn.Conv2d(512, 128, 1)

        # uncertainty
        self.uncer1=nn.Conv2d(320,1,1,1)
        self.uncer2=nn.Conv2d(320,1,1,1)
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

    def _forward(self, data):
        ## --------------------------Encoder--------------------------
        image = data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        # inputs 3*384*512
        h1 = self.block1(image)  # h1->64*384*512

        h2 = self.block2(h1)  # h2->128*192*256

        h3 = self.block3(h2)  # h3->256*96*128

        h4 = self.block4(h3)  # h4->512*48*64

        hd5 = self.block5(h4)  # h5->512*24*32

        ## --------------------------Decoder--------------------------
        # hd4->320*48*64
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        # hd3->320*96*128
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        # hd2->320*192*256
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))

        # hd1->320*384*512
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))

        feature_map_1 = self.outconv1(hd1)  # feature1->32*384*512
        feature_map_3 = self.outconv2(hd3)  # feature3->128*96*128
        feature_map_5 = self.outconv3(hd5)  # feature5->128*24*32

        uncertainty_map1=self.uncer1(hd1)   # un1->1*384*512
        uncertainty_map3=self.uncer2(hd3)   # un3->1*96*128
        uncertainty_map5=self.uncer3(hd5)   # un5->1*24*32

        out_features = []
        out_features.append(feature_map_1)
        out_features.append(feature_map_3)
        out_features.append(feature_map_5)
        pred = {'feature_maps': out_features}

        if self.conf.compute_uncertainty:
            confidences = []
            conf = torch.sigmoid(uncertainty_map1)
            confidences.append(conf)
            conf = torch.sigmoid(uncertainty_map3)
            confidences.append(conf)
            conf = torch.sigmoid(uncertainty_map5)
            confidences.append(conf)
        pred['confidences'] = confidences
        return pred


    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
    input = torch.randn((1, 3, 384, 512))
    model = UNet3Plus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input=input.cuda(device)
    # model.to(device)
    f1,f3,f5,u1,u3 = model(input)
    # print(f1,f3,f5,u1,u3,u5)
    print(f1.shape,f3.shape,f5.shape,u1.shape,u3.shape)