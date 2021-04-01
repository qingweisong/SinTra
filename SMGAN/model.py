import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18
import math
import SMGAN.relativeAttention as relativeAttention
from SMGAN.model_xl import *

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

    def __init__(self, d_model, dropout=0.2, max_len=5000):
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
        x = x + self.pe[:x.size(0), :] * 0.1
        return self.dropout(x)


class TransformerBlock(nn.Module):

    def __init__(
            self,
            track,
            nword,
            ninp = 32,
            nhead = 8,
            nhid = 512,
            nlayers = 6,
            dropout=0.2
    ):
        super(TransformerBlock, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.track = track
        self.ninp = ninp

        self.sine_position_encoder = PositionalEncoding(ninp)

        # embeding
        # 1st: pitch -> 64 -> 32 -> 1
        # 2rd: track -> feature
        self.embeding = nn.Embedding(nword, ninp)

        # transformer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # deocder
        self.decoder = nn.Linear(ninp, nword)

        # self.init_weights()

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

    def forward(self, img, draw_concat=False): # input: 1, track, length
        img = img.long().cuda()
        img = img.permute(0, 2, 1).reshape(1, -1) # 1, length*track

        seq_len = img.shape[1]

        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            device = img.device
            mask = self._generate_square_subsequent_mask(seq_len)#.to(device)
            mask = mask.to(device)
            self.src_mask = mask

        src = self.embeding(img).permute(1, 0, 2) # length, 1, feature

        src = src * math.sqrt(self.ninp)
        output = self.transformer_encoder(
            self.sine_position_encoder(src),
            self.src_mask
        ) # length, 1, feature


        output = self.decoder(output) # length(length*track), 1, nword

        if draw_concat == True:
            top1 = torch.zeros([output.shape[0], output.shape[1]]) # (time*track), 1
            top1 = output.argmax(dim=2).permute(1, 0) #  1, length*track
            top1 = top1.reshape([1, -1, self.track]) # 1, length, track
            top1 = top1.permute(0, 2, 1) # 1, track, length
            return top1 
        
        output = output.permute(1, 2, 0) # 1, nword, length
        output = output.reshape(output.shape[0], output.shape[1], -1, self.track)
        output = output.permute(0, 1, 3, 2)

        return output # 1, nword, track, length


# =================== Transformer END =====================================
class D_transform(nn.Module):
    def __init__(self, opt):
        super(D_transform, self).__init__()
        self.transformer = TransformerBlock(opt.ntrack, opt.nword)
    
    def forward(self,x, mems=None, draw_concat=False):
        x = self.transformer(x, draw_concat)
        return x, None


class G_transform(nn.Module):
    def __init__(self, opt):
        super(G_transform, self).__init__()
        self.transformer = TransformerBlock(opt.ntrack, opt.nword)
    
    def forward(self, x, y=None, mems=None, draw_concat=False):
        x = self.transformer(x, draw_concat)
        return x, None


class D_transformXL(nn.Module):
    def __init__(self, opt):
        super(D_transformXL, self).__init__()
        self.transformer = TransformerXL(
            opt.ntrack, 
            # n_layer, 
            # n_head, 
            # d_model, 
            # d_head, 
            # d_inner,
            # dropout, 
            # dropatt
            tgt_len=128,
            mem_len=128,
            ext_len=0,
            )


    def forward(self,x, mems=None):

        x, mems = self.transformer(x, mems)

        return x, mems


class G_transformXL(nn.Module):
    def __init__(self, opt):
        super(G_transformXL, self).__init__()
        self.transformer = TransformerXL(
            opt.ntrack, 
            # n_layer, 
            # n_head, 
            # d_model, 
            # d_head, 
            # d_inner,
            # dropout, 
            # dropatt.
            tgt_len=128,
            mem_len=128,
            ext_len=0,
            )

    def forward(self,x,y, mems=None):

        x, memes = self.transformer(x, mems)

        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]#??????????
        return x+y, memes



class WDiscriminator_init(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator_init, self).__init__()
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


class GeneratorConcatSkip2CleanAdd_init(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd_init, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
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
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y