# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class MobileNetV2(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MobileNetV2::input_0(MobileNetV2::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=28, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/Conv2dNormActivation[0]/Conv2d[0]/ret.3(MobileNetV2::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/Conv2dNormActivation[0]/ReLU[2]/10897(MobileNetV2::nndct_relu_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=28, out_channels=28, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=28, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[1]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.7(MobileNetV2::nndct_depthwise_conv2d_3)
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[1]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/10925(MobileNetV2::nndct_relu_4)
        self.module_5 = py_nndct.nn.Conv2d(in_channels=28, out_channels=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[1]/Sequential[conv]/Conv2d[1]/ret.11(MobileNetV2::nndct_conv2d_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=14, out_channels=72, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.15(MobileNetV2::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/10980(MobileNetV2::nndct_relu_7)
        self.module_8 = py_nndct.nn.Conv2d(in_channels=72, out_channels=72, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=72, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.19(MobileNetV2::nndct_depthwise_conv2d_8)
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11008(MobileNetV2::nndct_relu_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=72, out_channels=22, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/Conv2d[2]/ret.23(MobileNetV2::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.Conv2d(in_channels=22, out_channels=130, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.27(MobileNetV2::nndct_conv2d_11)
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11063(MobileNetV2::nndct_relu_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=130, out_channels=130, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=130, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.31(MobileNetV2::nndct_depthwise_conv2d_13)
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11091(MobileNetV2::nndct_relu_14)
        self.module_15 = py_nndct.nn.Conv2d(in_channels=130, out_channels=22, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/Sequential[conv]/Conv2d[2]/ret.35(MobileNetV2::nndct_conv2d_15)
        self.module_16 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[3]/ret.39(MobileNetV2::nndct_elemwise_add_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=22, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.41(MobileNetV2::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11149(MobileNetV2::nndct_relu_18)
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.45(MobileNetV2::nndct_depthwise_conv2d_19)
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11177(MobileNetV2::nndct_relu_20)
        self.module_21 = py_nndct.nn.Conv2d(in_channels=128, out_channels=30, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/Conv2d[2]/ret.49(MobileNetV2::nndct_conv2d_21)
        self.module_22 = py_nndct.nn.Conv2d(in_channels=30, out_channels=162, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.53(MobileNetV2::nndct_conv2d_22)
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11232(MobileNetV2::nndct_relu_23)
        self.module_24 = py_nndct.nn.Conv2d(in_channels=162, out_channels=162, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=162, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.57(MobileNetV2::nndct_depthwise_conv2d_24)
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11260(MobileNetV2::nndct_relu_25)
        self.module_26 = py_nndct.nn.Conv2d(in_channels=162, out_channels=30, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/Sequential[conv]/Conv2d[2]/ret.61(MobileNetV2::nndct_conv2d_26)
        self.module_27 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[5]/ret.65(MobileNetV2::nndct_elemwise_add_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=30, out_channels=86, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.67(MobileNetV2::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11318(MobileNetV2::nndct_relu_29)
        self.module_30 = py_nndct.nn.Conv2d(in_channels=86, out_channels=86, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=86, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.71(MobileNetV2::nndct_depthwise_conv2d_30)
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11346(MobileNetV2::nndct_relu_31)
        self.module_32 = py_nndct.nn.Conv2d(in_channels=86, out_channels=30, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/Sequential[conv]/Conv2d[2]/ret.75(MobileNetV2::nndct_conv2d_32)
        self.module_33 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[6]/ret.79(MobileNetV2::nndct_elemwise_add_33)
        self.module_34 = py_nndct.nn.Conv2d(in_channels=30, out_channels=182, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.81(MobileNetV2::nndct_conv2d_34)
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11404(MobileNetV2::nndct_relu_35)
        self.module_36 = py_nndct.nn.Conv2d(in_channels=182, out_channels=182, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=182, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.85(MobileNetV2::nndct_depthwise_conv2d_36)
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11432(MobileNetV2::nndct_relu_37)
        self.module_38 = py_nndct.nn.Conv2d(in_channels=182, out_channels=60, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/Conv2d[2]/ret.89(MobileNetV2::nndct_conv2d_38)
        self.module_39 = py_nndct.nn.Conv2d(in_channels=60, out_channels=316, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.93(MobileNetV2::nndct_conv2d_39)
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11487(MobileNetV2::nndct_relu_40)
        self.module_41 = py_nndct.nn.Conv2d(in_channels=316, out_channels=316, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=316, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.97(MobileNetV2::nndct_depthwise_conv2d_41)
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11515(MobileNetV2::nndct_relu_42)
        self.module_43 = py_nndct.nn.Conv2d(in_channels=316, out_channels=60, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/Conv2d[2]/ret.101(MobileNetV2::nndct_conv2d_43)
        self.module_44 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[8]/ret.105(MobileNetV2::nndct_elemwise_add_44)
        self.module_45 = py_nndct.nn.Conv2d(in_channels=60, out_channels=282, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.107(MobileNetV2::nndct_conv2d_45)
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11573(MobileNetV2::nndct_relu_46)
        self.module_47 = py_nndct.nn.Conv2d(in_channels=282, out_channels=282, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=282, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.111(MobileNetV2::nndct_depthwise_conv2d_47)
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11601(MobileNetV2::nndct_relu_48)
        self.module_49 = py_nndct.nn.Conv2d(in_channels=282, out_channels=60, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/Sequential[conv]/Conv2d[2]/ret.115(MobileNetV2::nndct_conv2d_49)
        self.module_50 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[9]/ret.119(MobileNetV2::nndct_elemwise_add_50)
        self.module_51 = py_nndct.nn.Conv2d(in_channels=60, out_channels=252, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.121(MobileNetV2::nndct_conv2d_51)
        self.module_52 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11659(MobileNetV2::nndct_relu_52)
        self.module_53 = py_nndct.nn.Conv2d(in_channels=252, out_channels=252, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=252, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.125(MobileNetV2::nndct_depthwise_conv2d_53)
        self.module_54 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11687(MobileNetV2::nndct_relu_54)
        self.module_55 = py_nndct.nn.Conv2d(in_channels=252, out_channels=60, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/Sequential[conv]/Conv2d[2]/ret.129(MobileNetV2::nndct_conv2d_55)
        self.module_56 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[10]/ret.133(MobileNetV2::nndct_elemwise_add_56)
        self.module_57 = py_nndct.nn.Conv2d(in_channels=60, out_channels=364, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.135(MobileNetV2::nndct_conv2d_57)
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11745(MobileNetV2::nndct_relu_58)
        self.module_59 = py_nndct.nn.Conv2d(in_channels=364, out_channels=364, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=364, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.139(MobileNetV2::nndct_depthwise_conv2d_59)
        self.module_60 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11773(MobileNetV2::nndct_relu_60)
        self.module_61 = py_nndct.nn.Conv2d(in_channels=364, out_channels=92, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/Conv2d[2]/ret.143(MobileNetV2::nndct_conv2d_61)
        self.module_62 = py_nndct.nn.Conv2d(in_channels=92, out_channels=548, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.147(MobileNetV2::nndct_conv2d_62)
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11828(MobileNetV2::nndct_relu_63)
        self.module_64 = py_nndct.nn.Conv2d(in_channels=548, out_channels=548, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=548, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.151(MobileNetV2::nndct_depthwise_conv2d_64)
        self.module_65 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11856(MobileNetV2::nndct_relu_65)
        self.module_66 = py_nndct.nn.Conv2d(in_channels=548, out_channels=92, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/Sequential[conv]/Conv2d[2]/ret.155(MobileNetV2::nndct_conv2d_66)
        self.module_67 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[12]/ret.159(MobileNetV2::nndct_elemwise_add_67)
        self.module_68 = py_nndct.nn.Conv2d(in_channels=92, out_channels=446, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.161(MobileNetV2::nndct_conv2d_68)
        self.module_69 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/11914(MobileNetV2::nndct_relu_69)
        self.module_70 = py_nndct.nn.Conv2d(in_channels=446, out_channels=446, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=446, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.165(MobileNetV2::nndct_depthwise_conv2d_70)
        self.module_71 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/11942(MobileNetV2::nndct_relu_71)
        self.module_72 = py_nndct.nn.Conv2d(in_channels=446, out_channels=92, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/Sequential[conv]/Conv2d[2]/ret.169(MobileNetV2::nndct_conv2d_72)
        self.module_73 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[13]/ret.173(MobileNetV2::nndct_elemwise_add_73)
        self.module_74 = py_nndct.nn.Conv2d(in_channels=92, out_channels=548, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.175(MobileNetV2::nndct_conv2d_74)
        self.module_75 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/12000(MobileNetV2::nndct_relu_75)
        self.module_76 = py_nndct.nn.Conv2d(in_channels=548, out_channels=548, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=548, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.179(MobileNetV2::nndct_depthwise_conv2d_76)
        self.module_77 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/12028(MobileNetV2::nndct_relu_77)
        self.module_78 = py_nndct.nn.Conv2d(in_channels=548, out_channels=146, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/Conv2d[2]/ret.183(MobileNetV2::nndct_conv2d_78)
        self.module_79 = py_nndct.nn.Conv2d(in_channels=146, out_channels=912, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.187(MobileNetV2::nndct_conv2d_79)
        self.module_80 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/12083(MobileNetV2::nndct_relu_80)
        self.module_81 = py_nndct.nn.Conv2d(in_channels=912, out_channels=912, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=912, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.191(MobileNetV2::nndct_depthwise_conv2d_81)
        self.module_82 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/12111(MobileNetV2::nndct_relu_82)
        self.module_83 = py_nndct.nn.Conv2d(in_channels=912, out_channels=146, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/Conv2d[2]/ret.195(MobileNetV2::nndct_conv2d_83)
        self.module_84 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[15]/ret.199(MobileNetV2::nndct_elemwise_add_84)
        self.module_85 = py_nndct.nn.Conv2d(in_channels=146, out_channels=912, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.201(MobileNetV2::nndct_conv2d_85)
        self.module_86 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/12169(MobileNetV2::nndct_relu_86)
        self.module_87 = py_nndct.nn.Conv2d(in_channels=912, out_channels=912, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=912, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.205(MobileNetV2::nndct_depthwise_conv2d_87)
        self.module_88 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/12197(MobileNetV2::nndct_relu_88)
        self.module_89 = py_nndct.nn.Conv2d(in_channels=912, out_channels=146, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/Conv2d[2]/ret.209(MobileNetV2::nndct_conv2d_89)
        self.module_90 = py_nndct.nn.Add() #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[16]/ret.213(MobileNetV2::nndct_elemwise_add_90)
        self.module_91 = py_nndct.nn.Conv2d(in_channels=146, out_channels=912, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/Conv2dNormActivation[0]/Conv2d[0]/ret.215(MobileNetV2::nndct_conv2d_91)
        self.module_92 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/Conv2dNormActivation[0]/ReLU[2]/12255(MobileNetV2::nndct_relu_92)
        self.module_93 = py_nndct.nn.Conv2d(in_channels=912, out_channels=912, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=912, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/Conv2dNormActivation[1]/Conv2d[0]/ret.219(MobileNetV2::nndct_depthwise_conv2d_93)
        self.module_94 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/Conv2dNormActivation[1]/ReLU[2]/12283(MobileNetV2::nndct_relu_94)
        self.module_95 = py_nndct.nn.Conv2d(in_channels=912, out_channels=304, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/InvertedResidual[17]/Sequential[conv]/Conv2d[2]/ret.223(MobileNetV2::nndct_conv2d_95)
        self.module_96 = py_nndct.nn.Conv2d(in_channels=304, out_channels=998, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV2::MobileNetV2/Sequential[features]/Conv2dNormActivation[18]/Conv2d[0]/ret.227(MobileNetV2::nndct_conv2d_96)
        self.module_97 = py_nndct.nn.ReLU(inplace=True) #MobileNetV2::MobileNetV2/Sequential[features]/Conv2dNormActivation[18]/ReLU[2]/12338(MobileNetV2::nndct_relu_97)
        self.module_98 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV2::MobileNetV2/12355(MobileNetV2::nndct_adaptive_avg_pool2d_98)
        self.module_99 = py_nndct.nn.Module('nndct_flatten') #MobileNetV2::MobileNetV2/ret.231(MobileNetV2::nndct_flatten_99)
        self.module_100 = py_nndct.nn.Linear(in_features=998, out_features=128, bias=True) #MobileNetV2::MobileNetV2/Sequential[classifier]/Linear[1]/ret.233(MobileNetV2::nndct_dense_100)
        self.module_101 = py_nndct.nn.ReLU(inplace=False) #MobileNetV2::MobileNetV2/Sequential[classifier]/ReLU[2]/ret.235(MobileNetV2::nndct_relu_101)
        self.module_102 = py_nndct.nn.Linear(in_features=128, out_features=10, bias=True) #MobileNetV2::MobileNetV2/Sequential[classifier]/Linear[3]/ret(MobileNetV2::nndct_dense_102)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_11 = self.module_11(output_module_0)
        output_module_11 = self.module_12(output_module_11)
        output_module_11 = self.module_13(output_module_11)
        output_module_11 = self.module_14(output_module_11)
        output_module_11 = self.module_15(output_module_11)
        output_module_16 = self.module_16(input=output_module_0, other=output_module_11, alpha=1)
        output_module_16 = self.module_17(output_module_16)
        output_module_16 = self.module_18(output_module_16)
        output_module_16 = self.module_19(output_module_16)
        output_module_16 = self.module_20(output_module_16)
        output_module_16 = self.module_21(output_module_16)
        output_module_22 = self.module_22(output_module_16)
        output_module_22 = self.module_23(output_module_22)
        output_module_22 = self.module_24(output_module_22)
        output_module_22 = self.module_25(output_module_22)
        output_module_22 = self.module_26(output_module_22)
        output_module_27 = self.module_27(input=output_module_16, other=output_module_22, alpha=1)
        output_module_28 = self.module_28(output_module_27)
        output_module_28 = self.module_29(output_module_28)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(output_module_28)
        output_module_28 = self.module_32(output_module_28)
        output_module_33 = self.module_33(input=output_module_27, other=output_module_28, alpha=1)
        output_module_33 = self.module_34(output_module_33)
        output_module_33 = self.module_35(output_module_33)
        output_module_33 = self.module_36(output_module_33)
        output_module_33 = self.module_37(output_module_33)
        output_module_33 = self.module_38(output_module_33)
        output_module_39 = self.module_39(output_module_33)
        output_module_39 = self.module_40(output_module_39)
        output_module_39 = self.module_41(output_module_39)
        output_module_39 = self.module_42(output_module_39)
        output_module_39 = self.module_43(output_module_39)
        output_module_44 = self.module_44(input=output_module_33, other=output_module_39, alpha=1)
        output_module_45 = self.module_45(output_module_44)
        output_module_45 = self.module_46(output_module_45)
        output_module_45 = self.module_47(output_module_45)
        output_module_45 = self.module_48(output_module_45)
        output_module_45 = self.module_49(output_module_45)
        output_module_50 = self.module_50(input=output_module_44, other=output_module_45, alpha=1)
        output_module_51 = self.module_51(output_module_50)
        output_module_51 = self.module_52(output_module_51)
        output_module_51 = self.module_53(output_module_51)
        output_module_51 = self.module_54(output_module_51)
        output_module_51 = self.module_55(output_module_51)
        output_module_56 = self.module_56(input=output_module_50, other=output_module_51, alpha=1)
        output_module_56 = self.module_57(output_module_56)
        output_module_56 = self.module_58(output_module_56)
        output_module_56 = self.module_59(output_module_56)
        output_module_56 = self.module_60(output_module_56)
        output_module_56 = self.module_61(output_module_56)
        output_module_62 = self.module_62(output_module_56)
        output_module_62 = self.module_63(output_module_62)
        output_module_62 = self.module_64(output_module_62)
        output_module_62 = self.module_65(output_module_62)
        output_module_62 = self.module_66(output_module_62)
        output_module_67 = self.module_67(input=output_module_56, other=output_module_62, alpha=1)
        output_module_68 = self.module_68(output_module_67)
        output_module_68 = self.module_69(output_module_68)
        output_module_68 = self.module_70(output_module_68)
        output_module_68 = self.module_71(output_module_68)
        output_module_68 = self.module_72(output_module_68)
        output_module_73 = self.module_73(input=output_module_67, other=output_module_68, alpha=1)
        output_module_73 = self.module_74(output_module_73)
        output_module_73 = self.module_75(output_module_73)
        output_module_73 = self.module_76(output_module_73)
        output_module_73 = self.module_77(output_module_73)
        output_module_73 = self.module_78(output_module_73)
        output_module_79 = self.module_79(output_module_73)
        output_module_79 = self.module_80(output_module_79)
        output_module_79 = self.module_81(output_module_79)
        output_module_79 = self.module_82(output_module_79)
        output_module_79 = self.module_83(output_module_79)
        output_module_84 = self.module_84(input=output_module_73, other=output_module_79, alpha=1)
        output_module_85 = self.module_85(output_module_84)
        output_module_85 = self.module_86(output_module_85)
        output_module_85 = self.module_87(output_module_85)
        output_module_85 = self.module_88(output_module_85)
        output_module_85 = self.module_89(output_module_85)
        output_module_90 = self.module_90(input=output_module_84, other=output_module_85, alpha=1)
        output_module_90 = self.module_91(output_module_90)
        output_module_90 = self.module_92(output_module_90)
        output_module_90 = self.module_93(output_module_90)
        output_module_90 = self.module_94(output_module_90)
        output_module_90 = self.module_95(output_module_90)
        output_module_90 = self.module_96(output_module_90)
        output_module_90 = self.module_97(output_module_90)
        output_module_90 = self.module_98(output_module_90)
        output_module_90 = self.module_99(input=output_module_90, start_dim=1, end_dim=-1)
        output_module_90 = self.module_100(output_module_90)
        output_module_90 = self.module_101(output_module_90)
        output_module_90 = self.module_102(output_module_90)
        return output_module_90
