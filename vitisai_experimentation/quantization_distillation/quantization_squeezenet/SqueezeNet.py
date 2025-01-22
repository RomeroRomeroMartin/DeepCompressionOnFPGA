# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class SqueezeNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SqueezeNet::input_0(SqueezeNet::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=[7, 7], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Conv2d[0]/ret.3(SqueezeNet::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/ReLU[1]/3349(SqueezeNet::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[2]/3364(SqueezeNet::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=96, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[squeeze]/ret.5(SqueezeNet::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[squeeze_activation]/3386(SqueezeNet::nndct_relu_5)
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[expand1x1]/ret.7(SqueezeNet::nndct_conv2d_6)
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[expand1x1_activation]/3407(SqueezeNet::nndct_relu_7)
        self.module_8 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[expand3x3]/ret.9(SqueezeNet::nndct_conv2d_8)
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[expand3x3_activation]/3428(SqueezeNet::nndct_relu_9)
        self.module_10 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ret.11(SqueezeNet::nndct_concat_10)
        self.module_11 = py_nndct.nn.Conv2d(in_channels=128, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[squeeze]/ret.13(SqueezeNet::nndct_conv2d_11)
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/3453(SqueezeNet::nndct_relu_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[expand1x1]/ret.15(SqueezeNet::nndct_conv2d_13)
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[expand1x1_activation]/3474(SqueezeNet::nndct_relu_14)
        self.module_15 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[expand3x3]/ret.17(SqueezeNet::nndct_conv2d_15)
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[expand3x3_activation]/3495(SqueezeNet::nndct_relu_16)
        self.module_17 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ret.19(SqueezeNet::nndct_concat_17)
        self.module_18 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/Conv2d[squeeze]/ret.21(SqueezeNet::nndct_conv2d_18)
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/ReLU[squeeze_activation]/3520(SqueezeNet::nndct_relu_19)
        self.module_20 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/Conv2d[expand1x1]/ret.23(SqueezeNet::nndct_conv2d_20)
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/ReLU[expand1x1_activation]/3541(SqueezeNet::nndct_relu_21)
        self.module_22 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/Conv2d[expand3x3]/ret.25(SqueezeNet::nndct_conv2d_22)
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/ReLU[expand3x3_activation]/3562(SqueezeNet::nndct_relu_23)
        self.module_24 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[5]/ret.27(SqueezeNet::nndct_concat_24)
        self.module_25 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[6]/3581(SqueezeNet::nndct_maxpool_25)
        self.module_26 = py_nndct.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[squeeze]/ret.29(SqueezeNet::nndct_conv2d_26)
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[squeeze_activation]/3603(SqueezeNet::nndct_relu_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[expand1x1]/ret.31(SqueezeNet::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[expand1x1_activation]/3624(SqueezeNet::nndct_relu_29)
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[expand3x3]/ret.33(SqueezeNet::nndct_conv2d_30)
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[expand3x3_activation]/3645(SqueezeNet::nndct_relu_31)
        self.module_32 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ret.35(SqueezeNet::nndct_concat_32)
        self.module_33 = py_nndct.nn.Conv2d(in_channels=256, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/Conv2d[squeeze]/ret.37(SqueezeNet::nndct_conv2d_33)
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/ReLU[squeeze_activation]/3670(SqueezeNet::nndct_relu_34)
        self.module_35 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/Conv2d[expand1x1]/ret.39(SqueezeNet::nndct_conv2d_35)
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/ReLU[expand1x1_activation]/3691(SqueezeNet::nndct_relu_36)
        self.module_37 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/Conv2d[expand3x3]/ret.41(SqueezeNet::nndct_conv2d_37)
        self.module_38 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/ReLU[expand3x3_activation]/3712(SqueezeNet::nndct_relu_38)
        self.module_39 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[8]/ret.43(SqueezeNet::nndct_concat_39)
        self.module_40 = py_nndct.nn.Conv2d(in_channels=384, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[squeeze]/ret.45(SqueezeNet::nndct_conv2d_40)
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/3737(SqueezeNet::nndct_relu_41)
        self.module_42 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[expand1x1]/ret.47(SqueezeNet::nndct_conv2d_42)
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[expand1x1_activation]/3758(SqueezeNet::nndct_relu_43)
        self.module_44 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[expand3x3]/ret.49(SqueezeNet::nndct_conv2d_44)
        self.module_45 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[expand3x3_activation]/3779(SqueezeNet::nndct_relu_45)
        self.module_46 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ret.51(SqueezeNet::nndct_concat_46)
        self.module_47 = py_nndct.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[squeeze]/ret.53(SqueezeNet::nndct_conv2d_47)
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[squeeze_activation]/3804(SqueezeNet::nndct_relu_48)
        self.module_49 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[expand1x1]/ret.55(SqueezeNet::nndct_conv2d_49)
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[expand1x1_activation]/3825(SqueezeNet::nndct_relu_50)
        self.module_51 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[expand3x3]/ret.57(SqueezeNet::nndct_conv2d_51)
        self.module_52 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[expand3x3_activation]/3846(SqueezeNet::nndct_relu_52)
        self.module_53 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ret.59(SqueezeNet::nndct_concat_53)
        self.module_54 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[11]/3865(SqueezeNet::nndct_maxpool_54)
        self.module_55 = py_nndct.nn.Conv2d(in_channels=512, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[squeeze]/ret.61(SqueezeNet::nndct_conv2d_55)
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[squeeze_activation]/3887(SqueezeNet::nndct_relu_56)
        self.module_57 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[expand1x1]/ret.63(SqueezeNet::nndct_conv2d_57)
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[expand1x1_activation]/3908(SqueezeNet::nndct_relu_58)
        self.module_59 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[expand3x3]/ret.65(SqueezeNet::nndct_conv2d_59)
        self.module_60 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[expand3x3_activation]/3929(SqueezeNet::nndct_relu_60)
        self.module_61 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ret.67(SqueezeNet::nndct_concat_61)
        self.module_62 = py_nndct.nn.Conv2d(in_channels=512, out_channels=10, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[classifier]/Conv2d[1]/ret.69(SqueezeNet::nndct_conv2d_62)
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[classifier]/ReLU[2]/3957(SqueezeNet::nndct_relu_63)
        self.module_64 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #SqueezeNet::SqueezeNet/Sequential[classifier]/AdaptiveAvgPool2d[3]/3974(SqueezeNet::nndct_adaptive_avg_pool2d_64)
        self.module_65 = py_nndct.nn.Module('nndct_flatten') #SqueezeNet::SqueezeNet/ret(SqueezeNet::nndct_flatten_65)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_6 = self.module_6(output_module_0)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(output_module_0)
        output_module_8 = self.module_9(output_module_8)
        output_module_6 = self.module_10(dim=1, tensors=[output_module_6,output_module_8])
        output_module_6 = self.module_11(output_module_6)
        output_module_6 = self.module_12(output_module_6)
        output_module_13 = self.module_13(output_module_6)
        output_module_13 = self.module_14(output_module_13)
        output_module_15 = self.module_15(output_module_6)
        output_module_15 = self.module_16(output_module_15)
        output_module_13 = self.module_17(dim=1, tensors=[output_module_13,output_module_15])
        output_module_13 = self.module_18(output_module_13)
        output_module_13 = self.module_19(output_module_13)
        output_module_20 = self.module_20(output_module_13)
        output_module_20 = self.module_21(output_module_20)
        output_module_22 = self.module_22(output_module_13)
        output_module_22 = self.module_23(output_module_22)
        output_module_20 = self.module_24(dim=1, tensors=[output_module_20,output_module_22])
        output_module_20 = self.module_25(output_module_20)
        output_module_20 = self.module_26(output_module_20)
        output_module_20 = self.module_27(output_module_20)
        output_module_28 = self.module_28(output_module_20)
        output_module_28 = self.module_29(output_module_28)
        output_module_30 = self.module_30(output_module_20)
        output_module_30 = self.module_31(output_module_30)
        output_module_28 = self.module_32(dim=1, tensors=[output_module_28,output_module_30])
        output_module_28 = self.module_33(output_module_28)
        output_module_28 = self.module_34(output_module_28)
        output_module_35 = self.module_35(output_module_28)
        output_module_35 = self.module_36(output_module_35)
        output_module_37 = self.module_37(output_module_28)
        output_module_37 = self.module_38(output_module_37)
        output_module_35 = self.module_39(dim=1, tensors=[output_module_35,output_module_37])
        output_module_35 = self.module_40(output_module_35)
        output_module_35 = self.module_41(output_module_35)
        output_module_42 = self.module_42(output_module_35)
        output_module_42 = self.module_43(output_module_42)
        output_module_44 = self.module_44(output_module_35)
        output_module_44 = self.module_45(output_module_44)
        output_module_42 = self.module_46(dim=1, tensors=[output_module_42,output_module_44])
        output_module_42 = self.module_47(output_module_42)
        output_module_42 = self.module_48(output_module_42)
        output_module_49 = self.module_49(output_module_42)
        output_module_49 = self.module_50(output_module_49)
        output_module_51 = self.module_51(output_module_42)
        output_module_51 = self.module_52(output_module_51)
        output_module_49 = self.module_53(dim=1, tensors=[output_module_49,output_module_51])
        output_module_49 = self.module_54(output_module_49)
        output_module_49 = self.module_55(output_module_49)
        output_module_49 = self.module_56(output_module_49)
        output_module_57 = self.module_57(output_module_49)
        output_module_57 = self.module_58(output_module_57)
        output_module_59 = self.module_59(output_module_49)
        output_module_59 = self.module_60(output_module_59)
        output_module_57 = self.module_61(dim=1, tensors=[output_module_57,output_module_59])
        output_module_57 = self.module_62(output_module_57)
        output_module_57 = self.module_63(output_module_57)
        output_module_57 = self.module_64(output_module_57)
        output_module_57 = self.module_65(input=output_module_57, start_dim=1, end_dim=-1)
        return output_module_57
