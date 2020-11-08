import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#卷积块
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.ntrack,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc#32 out_channel
        self.head = ConvBlock(opt.ntrack,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.ntrack,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]#??????????
        return x+y




# class bar_main(nn.Sequential):
#     def __init__(self, in_channel, out_channel):
#         super(ConvBlock,self).__init__()
#         self.add_module('transconv3d',nn.ConvTranspose3d(in_channel, 512, (1,4,1), (1,4,1), 0)),
#         self.add_module('norm',nn.BatchNorm2d(512)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),
#         self.add_module('transconv3d',nn.ConvTranspose3d(512, 256, (1,1,3), (1,1,3), 0)),
#         self.add_module('norm',nn.BatchNorm2d(256)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),
#         self.add_module('transconv3d',nn.ConvTranspose3d(256, 128, (1,4,1), (1,4,1), 0))
#         self.add_module('norm',nn.BatchNorm2d(128)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),
#         self.add_module('transconv3d',nn.ConvTranspose3d(128, out_channel, (1,1,3), (1,1,2), 0)),
#         self.add_module('norm',nn.BatchNorm2d(out_channel)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),

# def weights_init(m):
    

# class Generator_temp(nn.Module):
#     def __init__(self, opt):
#         super(Generator_temp, self).__init__()
#         self.dense = nn.Linear(32, 768)
#         self.bn1 = nn.BatchNorm2d(768)#output channel=32
#         #self.reshape1 = torch.reshape(32, 3, 1, 1, 256)
#         self.transconv3d = nn.ConvTranspose3d(256, 32, (2, 1, 1), 1, 0)#output= input*kernerl_size - (kernel_size - stride)*(input - 1)
#         self.bn2 = nn.BatchNorm2d(32)

#     def forward(self, input):#(32, 32)
#         x = F.leaky_relu(self.bn1(self.dense(input)))
#         x = x.reshape(32, 3, 1, 1, 256)
#         x = F.leaky_relu(self.bn2(self.transconv3d(x)))
#         x = x.reshape(32, 4, 32)

#         return x

# class Generator_bar(nn.Module):
#     def __init__(self, opt):
#         super(Generator, self).__init__()

#         self.head = bar_main(128, 64)

#         self.body1 = nn.Sequential()#bar_pitch_time
#         self.add_module('transconv3d',nn.ConvTranspose3d(64, 32, (1,1,12), (1,1,12), 0)),
#         self.add_module('norm',nn.BatchNorm2d(32)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),

#         self.add_module('transconv3d', nn.ConvTranspose3d(32, 16, (1,6,1), (1,6,1), 0)),
#         self.add_module('norm', nn.BatchNorm2d(16)),
#         self.add_module('leaky_relu', F.leaky_relu(0.2)),

#         self.body2 = nn.Sequential()#bar_time_pitch
#         self.add_module('transconv3d',nn.ConvTranspose3d(64, 32, (1,6,1), (1,6,1), 0)),
#         self.add_module('norm',nn.BatchNorm2d(32)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),

#         self.add_module('transconv3d',nn.ConvTranspose3d(32, 16, (1,1,12), (1,1,12), 0)),
#         self.add_module('norm',nn.BatchNorm2d(16)),
#         self.add_module('leaky_relu',F.leaky_relu(0.2)),

#         self.tail = nn.Sequential()#bar_merged
#         self.add_module('transconv3d',nn.ConvTranspose3d(32, 1, (1,1,1), (1,1,1), 0)),
#         self.add_module('norm',nn.BatchNorm2d(1)),
#         self.add_module('sigmoid', F.sigmoid()),

#     def forward(self, input):
#         x = input.reshape(32, 4, 1, 1, 128)
#         x = self.head(x)
#         x1 = self.body1(x)
#         x2 = self.body2(x)
#         x = self.tail(x1+x2)

#         return x

# class Discriminator(nn.Module):
#     def __init__(self, opt):
#         super(Discriminator, self).__init__()

#     def forward(self, x):

#         return x
