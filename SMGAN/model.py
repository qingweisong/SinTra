import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18
import math

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
   
# =================== Transformer START =========================

# class PositionEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """
#     def __init__(self, num_pos_feats=256):
#         super().__init__()
#         self.row_embed = nn.Embedding(50, num_pos_feats)
#         self.col_embed = nn.Embedding(50, num_pos_feats)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.uniform_(self.row_embed.weight)
#         nn.init.uniform_(self.col_embed.weight)

#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):

    def __init__(
            self,
            track = 7,
            ninp = 32,
            nhead = 4,
            nhid = 256,
            nlayers = 3,
            dropout=0.5
    ):
        super(TransformerBlock, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(128, ninp // 2))
        self.col_embed = nn.Parameter(torch.rand(512, ninp // 2))

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.embeding = nn.Conv2d(track, ninp, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.deconv = nn.ConvTranspose2d(ninp, ninp, 4, 2, 1, 0)
        self.ninp = ninp
        self.decoder = nn.Conv2d(ninp, track, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src):
    #     if self.src_mask is None or self.src_mask.size(0) != len(src):
    #         device = src.device
    #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
    #         self.src_mask = mask

    #     src = self.encoder(src) * math.sqrt(self.ninp)
    #     src = self.pos_encoder(src)
    #     output = self.transformer_encoder(src, self.src_mask)
    #     output = self.decoder(output)
    #     return output

    def forward(self, img):
        bilinear2d = nn.UpsamplingBilinear2d(img.shape[-2:])

        src = self.embeding(img)
        src = self.pool(src)

        originShape = src.shape

        H, W = originShape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        if self.src_mask is None or self.src_mask.size(0) != H*W:
            device = src.device
            mask = self._generate_square_subsequent_mask(H*W).to(device)
            self.src_mask = mask

        # import ipdb; ipdb.set_trace()

        output = self.transformer_encoder(
            pos * 0.05 + src.flatten(2).permute(2, 0, 1),
            self.src_mask
        ).permute(1, 2, 0)

        output = output.reshape(originShape)

        output = self.deconv(output)
        output = bilinear2d(output)

        output = self.decoder(output)

        return output


# =================== Transformer END =====================================

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        # N = int(opt.nfc)
        # self.head = ConvBlock(opt.ntrack,N,opt.ker_size,opt.padd_size,1)
        # self.body = nn.Sequential()
        # for i in range(opt.num_layer-2):
        #     N = int(opt.nfc/pow(2,(i+1)))
        #     block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
        #     self.body.add_module('block%d'%(i+1),block)
        # self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

        self.transformer = TransformerBlock(opt.ntrack)

    def forward(self,x):
        # x = self.head(x)
        # x = self.body(x)
        # x = self.tail(x)

        x = self.transformer(x)

        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        # N = opt.nfc#32 out_channel
        # self.head = ConvBlock(opt.ntrack,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        # self.body = nn.Sequential()
        # for i in range(opt.num_layer-2):
        #     N = int(opt.nfc/pow(2,(i+1)))
        #     block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
        #     self.body.add_module('block%d'%(i+1),block)
        # self.tail = nn.Sequential(
        #     nn.Conv2d(max(N,opt.min_nfc),opt.ntrack,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
        #     nn.Tanh()
        # )

        self.transformer = TransformerBlock(opt.ntrack)

    def forward(self,x,y):
        # x = self.head(x)
        # x = self.body(x)
        # x = self.tail(x)

        x = self.transformer(x)

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
