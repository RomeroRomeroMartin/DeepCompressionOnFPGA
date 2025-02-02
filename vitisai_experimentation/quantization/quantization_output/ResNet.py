# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0(ResNet::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/ret.3(ResNet::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/8911(ResNet::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #ResNet::ResNet/MaxPool2d[maxpool]/8926(ResNet::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]/ret.7(ResNet::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]/8955(ResNet::nndct_relu_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]/ret.11(ResNet::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]/8983(ResNet::nndct_relu_7)
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]/ret.15(ResNet::nndct_conv2d_8)
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/ret.19(ResNet::nndct_conv2d_9)
        self.module_10 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/9039(ResNet::nndct_elemwise_add_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu]/9040(ResNet::nndct_relu_11)
        self.module_12 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]/ret.23(ResNet::nndct_conv2d_12)
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/9068(ResNet::nndct_relu_13)
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]/ret.27(ResNet::nndct_conv2d_14)
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/9096(ResNet::nndct_relu_15)
        self.module_16 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]/ret.31(ResNet::nndct_conv2d_16)
        self.module_17 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/9125(ResNet::nndct_elemwise_add_17)
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/9126(ResNet::nndct_relu_18)
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]/ret.35(ResNet::nndct_conv2d_19)
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/9154(ResNet::nndct_relu_20)
        self.module_21 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]/ret.39(ResNet::nndct_conv2d_21)
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/9182(ResNet::nndct_relu_22)
        self.module_23 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]/ret.43(ResNet::nndct_conv2d_23)
        self.module_24 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/9211(ResNet::nndct_elemwise_add_24)
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/9212(ResNet::nndct_relu_25)
        self.module_26 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]/ret.47(ResNet::nndct_conv2d_26)
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]/9240(ResNet::nndct_relu_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]/ret.51(ResNet::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]/9268(ResNet::nndct_relu_29)
        self.module_30 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]/ret.55(ResNet::nndct_conv2d_30)
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/ret.59(ResNet::nndct_conv2d_31)
        self.module_32 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/9324(ResNet::nndct_elemwise_add_32)
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu]/9325(ResNet::nndct_relu_33)
        self.module_34 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]/ret.63(ResNet::nndct_conv2d_34)
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]/9353(ResNet::nndct_relu_35)
        self.module_36 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]/ret.67(ResNet::nndct_conv2d_36)
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]/9381(ResNet::nndct_relu_37)
        self.module_38 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]/ret.71(ResNet::nndct_conv2d_38)
        self.module_39 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/9410(ResNet::nndct_elemwise_add_39)
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu]/9411(ResNet::nndct_relu_40)
        self.module_41 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]/ret.75(ResNet::nndct_conv2d_41)
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]/9439(ResNet::nndct_relu_42)
        self.module_43 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]/ret.79(ResNet::nndct_conv2d_43)
        self.module_44 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]/9467(ResNet::nndct_relu_44)
        self.module_45 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]/ret.83(ResNet::nndct_conv2d_45)
        self.module_46 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/9496(ResNet::nndct_elemwise_add_46)
        self.module_47 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu]/9497(ResNet::nndct_relu_47)
        self.module_48 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]/ret.87(ResNet::nndct_conv2d_48)
        self.module_49 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]/9525(ResNet::nndct_relu_49)
        self.module_50 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]/ret.91(ResNet::nndct_conv2d_50)
        self.module_51 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]/9553(ResNet::nndct_relu_51)
        self.module_52 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]/ret.95(ResNet::nndct_conv2d_52)
        self.module_53 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/9582(ResNet::nndct_elemwise_add_53)
        self.module_54 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu]/9583(ResNet::nndct_relu_54)
        self.module_55 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]/ret.99(ResNet::nndct_conv2d_55)
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]/9611(ResNet::nndct_relu_56)
        self.module_57 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]/ret.103(ResNet::nndct_conv2d_57)
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]/9639(ResNet::nndct_relu_58)
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]/ret.107(ResNet::nndct_conv2d_59)
        self.module_60 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/ret.111(ResNet::nndct_conv2d_60)
        self.module_61 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/9695(ResNet::nndct_elemwise_add_61)
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu]/9696(ResNet::nndct_relu_62)
        self.module_63 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]/ret.115(ResNet::nndct_conv2d_63)
        self.module_64 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]/9724(ResNet::nndct_relu_64)
        self.module_65 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]/ret.119(ResNet::nndct_conv2d_65)
        self.module_66 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]/9752(ResNet::nndct_relu_66)
        self.module_67 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]/ret.123(ResNet::nndct_conv2d_67)
        self.module_68 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/9781(ResNet::nndct_elemwise_add_68)
        self.module_69 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu]/9782(ResNet::nndct_relu_69)
        self.module_70 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]/ret.127(ResNet::nndct_conv2d_70)
        self.module_71 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]/9810(ResNet::nndct_relu_71)
        self.module_72 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]/ret.131(ResNet::nndct_conv2d_72)
        self.module_73 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]/9838(ResNet::nndct_relu_73)
        self.module_74 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]/ret.135(ResNet::nndct_conv2d_74)
        self.module_75 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/9867(ResNet::nndct_elemwise_add_75)
        self.module_76 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu]/9868(ResNet::nndct_relu_76)
        self.module_77 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]/ret.139(ResNet::nndct_conv2d_77)
        self.module_78 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]/9896(ResNet::nndct_relu_78)
        self.module_79 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]/ret.143(ResNet::nndct_conv2d_79)
        self.module_80 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]/9924(ResNet::nndct_relu_80)
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]/ret.147(ResNet::nndct_conv2d_81)
        self.module_82 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/9953(ResNet::nndct_elemwise_add_82)
        self.module_83 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu]/9954(ResNet::nndct_relu_83)
        self.module_84 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]/ret.151(ResNet::nndct_conv2d_84)
        self.module_85 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]/9982(ResNet::nndct_relu_85)
        self.module_86 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]/ret.155(ResNet::nndct_conv2d_86)
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]/10010(ResNet::nndct_relu_87)
        self.module_88 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]/ret.159(ResNet::nndct_conv2d_88)
        self.module_89 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/10039(ResNet::nndct_elemwise_add_89)
        self.module_90 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu]/10040(ResNet::nndct_relu_90)
        self.module_91 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]/ret.163(ResNet::nndct_conv2d_91)
        self.module_92 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]/10068(ResNet::nndct_relu_92)
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]/ret.167(ResNet::nndct_conv2d_93)
        self.module_94 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]/10096(ResNet::nndct_relu_94)
        self.module_95 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]/ret.171(ResNet::nndct_conv2d_95)
        self.module_96 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/10125(ResNet::nndct_elemwise_add_96)
        self.module_97 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu]/10126(ResNet::nndct_relu_97)
        self.module_98 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]/ret.175(ResNet::nndct_conv2d_98)
        self.module_99 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]/10154(ResNet::nndct_relu_99)
        self.module_100 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]/ret.179(ResNet::nndct_conv2d_100)
        self.module_101 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]/10182(ResNet::nndct_relu_101)
        self.module_102 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]/ret.183(ResNet::nndct_conv2d_102)
        self.module_103 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/ret.187(ResNet::nndct_conv2d_103)
        self.module_104 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/10238(ResNet::nndct_elemwise_add_104)
        self.module_105 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu]/10239(ResNet::nndct_relu_105)
        self.module_106 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]/ret.191(ResNet::nndct_conv2d_106)
        self.module_107 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]/10267(ResNet::nndct_relu_107)
        self.module_108 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]/ret.195(ResNet::nndct_conv2d_108)
        self.module_109 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]/10295(ResNet::nndct_relu_109)
        self.module_110 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]/ret.199(ResNet::nndct_conv2d_110)
        self.module_111 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/10324(ResNet::nndct_elemwise_add_111)
        self.module_112 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]/10325(ResNet::nndct_relu_112)
        self.module_113 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]/ret.203(ResNet::nndct_conv2d_113)
        self.module_114 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]/10353(ResNet::nndct_relu_114)
        self.module_115 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]/ret.207(ResNet::nndct_conv2d_115)
        self.module_116 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]/10381(ResNet::nndct_relu_116)
        self.module_117 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]/ret.211(ResNet::nndct_conv2d_117)
        self.module_118 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/10410(ResNet::nndct_elemwise_add_118)
        self.module_119 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]/10411(ResNet::nndct_relu_119)
        self.module_120 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/10428(ResNet::nndct_adaptive_avg_pool2d_120)
        self.module_121 = py_nndct.nn.Module('nndct_flatten') #ResNet::ResNet/ret.215(ResNet::nndct_flatten_121)
        self.module_122 = py_nndct.nn.Linear(in_features=2048, out_features=256, bias=True) #ResNet::ResNet/Sequential[fc]/Linear[0]/ret.217(ResNet::nndct_dense_122)
        self.module_123 = py_nndct.nn.ReLU(inplace=False) #ResNet::ResNet/Sequential[fc]/ReLU[1]/ret.219(ResNet::nndct_relu_123)
        self.module_124 = py_nndct.nn.Linear(in_features=256, out_features=10, bias=True) #ResNet::ResNet/Sequential[fc]/Linear[3]/ret(ResNet::nndct_dense_124)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(output_module_4)
        output_module_4 = self.module_8(output_module_4)
        output_module_9 = self.module_9(output_module_0)
        output_module_4 = self.module_10(input=output_module_4, other=output_module_9, alpha=1)
        output_module_4 = self.module_11(output_module_4)
        output_module_12 = self.module_12(output_module_4)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_16(output_module_12)
        output_module_12 = self.module_17(input=output_module_12, other=output_module_4, alpha=1)
        output_module_12 = self.module_18(output_module_12)
        output_module_19 = self.module_19(output_module_12)
        output_module_19 = self.module_20(output_module_19)
        output_module_19 = self.module_21(output_module_19)
        output_module_19 = self.module_22(output_module_19)
        output_module_19 = self.module_23(output_module_19)
        output_module_19 = self.module_24(input=output_module_19, other=output_module_12, alpha=1)
        output_module_19 = self.module_25(output_module_19)
        output_module_26 = self.module_26(output_module_19)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_26 = self.module_30(output_module_26)
        output_module_31 = self.module_31(output_module_19)
        output_module_26 = self.module_32(input=output_module_26, other=output_module_31, alpha=1)
        output_module_26 = self.module_33(output_module_26)
        output_module_34 = self.module_34(output_module_26)
        output_module_34 = self.module_35(output_module_34)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_34 = self.module_38(output_module_34)
        output_module_34 = self.module_39(input=output_module_34, other=output_module_26, alpha=1)
        output_module_34 = self.module_40(output_module_34)
        output_module_41 = self.module_41(output_module_34)
        output_module_41 = self.module_42(output_module_41)
        output_module_41 = self.module_43(output_module_41)
        output_module_41 = self.module_44(output_module_41)
        output_module_41 = self.module_45(output_module_41)
        output_module_41 = self.module_46(input=output_module_41, other=output_module_34, alpha=1)
        output_module_41 = self.module_47(output_module_41)
        output_module_48 = self.module_48(output_module_41)
        output_module_48 = self.module_49(output_module_48)
        output_module_48 = self.module_50(output_module_48)
        output_module_48 = self.module_51(output_module_48)
        output_module_48 = self.module_52(output_module_48)
        output_module_48 = self.module_53(input=output_module_48, other=output_module_41, alpha=1)
        output_module_48 = self.module_54(output_module_48)
        output_module_55 = self.module_55(output_module_48)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_55 = self.module_59(output_module_55)
        output_module_60 = self.module_60(output_module_48)
        output_module_55 = self.module_61(input=output_module_55, other=output_module_60, alpha=1)
        output_module_55 = self.module_62(output_module_55)
        output_module_63 = self.module_63(output_module_55)
        output_module_63 = self.module_64(output_module_63)
        output_module_63 = self.module_65(output_module_63)
        output_module_63 = self.module_66(output_module_63)
        output_module_63 = self.module_67(output_module_63)
        output_module_63 = self.module_68(input=output_module_63, other=output_module_55, alpha=1)
        output_module_63 = self.module_69(output_module_63)
        output_module_70 = self.module_70(output_module_63)
        output_module_70 = self.module_71(output_module_70)
        output_module_70 = self.module_72(output_module_70)
        output_module_70 = self.module_73(output_module_70)
        output_module_70 = self.module_74(output_module_70)
        output_module_70 = self.module_75(input=output_module_70, other=output_module_63, alpha=1)
        output_module_70 = self.module_76(output_module_70)
        output_module_77 = self.module_77(output_module_70)
        output_module_77 = self.module_78(output_module_77)
        output_module_77 = self.module_79(output_module_77)
        output_module_77 = self.module_80(output_module_77)
        output_module_77 = self.module_81(output_module_77)
        output_module_77 = self.module_82(input=output_module_77, other=output_module_70, alpha=1)
        output_module_77 = self.module_83(output_module_77)
        output_module_84 = self.module_84(output_module_77)
        output_module_84 = self.module_85(output_module_84)
        output_module_84 = self.module_86(output_module_84)
        output_module_84 = self.module_87(output_module_84)
        output_module_84 = self.module_88(output_module_84)
        output_module_84 = self.module_89(input=output_module_84, other=output_module_77, alpha=1)
        output_module_84 = self.module_90(output_module_84)
        output_module_91 = self.module_91(output_module_84)
        output_module_91 = self.module_92(output_module_91)
        output_module_91 = self.module_93(output_module_91)
        output_module_91 = self.module_94(output_module_91)
        output_module_91 = self.module_95(output_module_91)
        output_module_91 = self.module_96(input=output_module_91, other=output_module_84, alpha=1)
        output_module_91 = self.module_97(output_module_91)
        output_module_98 = self.module_98(output_module_91)
        output_module_98 = self.module_99(output_module_98)
        output_module_98 = self.module_100(output_module_98)
        output_module_98 = self.module_101(output_module_98)
        output_module_98 = self.module_102(output_module_98)
        output_module_103 = self.module_103(output_module_91)
        output_module_98 = self.module_104(input=output_module_98, other=output_module_103, alpha=1)
        output_module_98 = self.module_105(output_module_98)
        output_module_106 = self.module_106(output_module_98)
        output_module_106 = self.module_107(output_module_106)
        output_module_106 = self.module_108(output_module_106)
        output_module_106 = self.module_109(output_module_106)
        output_module_106 = self.module_110(output_module_106)
        output_module_106 = self.module_111(input=output_module_106, other=output_module_98, alpha=1)
        output_module_106 = self.module_112(output_module_106)
        output_module_113 = self.module_113(output_module_106)
        output_module_113 = self.module_114(output_module_113)
        output_module_113 = self.module_115(output_module_113)
        output_module_113 = self.module_116(output_module_113)
        output_module_113 = self.module_117(output_module_113)
        output_module_113 = self.module_118(input=output_module_113, other=output_module_106, alpha=1)
        output_module_113 = self.module_119(output_module_113)
        output_module_113 = self.module_120(output_module_113)
        output_module_113 = self.module_121(input=output_module_113, start_dim=1, end_dim=-1)
        output_module_113 = self.module_122(output_module_113)
        output_module_113 = self.module_123(output_module_113)
        output_module_113 = self.module_124(output_module_113)
        return output_module_113
