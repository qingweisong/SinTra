import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18
import math
from SMGAN.relativeAttention import *


# =============================
# new GAN
# =============================

class G_noise(nn.Module):

    def __init__(self, opt):
        super(G_noise, self).__init__()
        self.ntrack = opt.ntrack
        self.nword = opt.nword
        
        feature_default = 64
        ninp = 64
        nhead = 8
        nhid = 512
        nlayers = 6
        dropout=0.1
        max_len=2048

        self.ninp = ninp
        self.nlayers = nlayers

        self.fcs = nn.ModuleList([nn.Linear(feature_default, ninp) for _ in range(self.ntrack)])

        self.pos_encoding = DynamicPositionEmbedding(ninp, max_seq=max_len)

        # transformer rga
        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(ninp, dropout, h=ninp // 16, additional=False, max_seq=max_len)
             for _ in range(nlayers)])
        self.dropout = torch.nn.Dropout(dropout)

        # handle track dim
        self.track_conv1 = nn.Conv2d(self.ntrack, 16, 1)
        self.track_conv2 = nn.Conv2d(16, 1, 1)
        self.track_deconv1 = nn.Conv2d(1, 16, 1)
        self.track_deconv2 = nn.Conv2d(16, self.ntrack, 1)

        # deocder
        self.decoder = nn.Linear(ninp, self.nword)
        self.decoders = nn.ModuleList([nn.Linear(ninp, self.nword) for i in range(self.ntrack)])

    def topP_sampling(self, x, p=0.8): # 1, track, length, nword
        xp = F.softmax(x, dim=-1) # 1, track, length, nword
        topP, indices = torch.sort(x, dim=-1, descending=True)
        cumsum = torch.cumsum(topP, dim=-1)
        mask = torch.where(cumsum < p, topP, torch.ones_like(topP)*1000)
        minP, indices = torch.min(mask, dim=-1, keepdim=True)
        valid_p = torch.where(xp<minP, torch.ones_like(x)*(1e-10), xp)
        sample = torch.distributions.Categorical(valid_p).sample()
        return sample # 1, track, lengtha
    
    def forward(self, noise, mems=None, mode="loss"): # [1, track, length, feature_default] float

        print("G_noise noise: ", noise.shape)
        
        tmp = []
        for i in range(noise.shape[1]):
            a_track = self.fcs[i](noise[:, i, :, :]) # 1, length, ninp
            a_track = a_track[:, None, :, :] # 1, 1, length, ninp
            tmp.append(a_track)
        src = torch.cat(tmp, dim=1) # 1, track, length, ninp

        src = self.track_conv1(src) # 1, 16, length, feature
        src = self.track_conv2(src) # 1, 1, length, feature
        src = src[0, :, :, :] # 1, length, feature

        tmp2 = noise[:, 0, :, 0] # 1, length
        print("G_noise tmp2: ", tmp2.shape)
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(tmp2.shape[1], tmp2, tmp2, -10)

        src = src * math.sqrt(self.ninp)

        weights = []
        x = src
        for i in range(self.nlayers):
            x, w = self.enc_layers[i](x, look_ahead_mask)
            weights.append(w)

        output = x[:, None, :, :] # 1, 1, length, feature
        output = self.track_deconv1(output) # 1, 16, length, feature
        output = self.track_deconv2(output) # 1, track, length, feature
        output_feature = output

        outputs = []
        for i in range(self.ntrack):
            a_track = self.decoders[i](output[:, i, :, :]) # 1, length, nword
            a_track = a_track[:, None, :, :] # 1, 1, length, nword
            outputs.append(a_track)

        output = torch.cat(outputs, dim=1) # 1, track, length, nword

        if mode == "feature":
            # top1 = output # 1, track, length, nword
            # top1 = top1.argmax(dim=3) # 1, track, length
            return output_feature, None #[1, track, length, feature]
        elif mode == "topP":
            sample = self.topP_sampling(output) # 1, track, length
            return sample, None
        elif mode == "top1":
            top1 = output # 1, track, length, nword
            top1 = top1.argmax(dim=3) # 1, track, length
            return top1, None
        elif mode == "loss":
            output = output.permute(0, 3, 1, 2) # 1, nword, track, length
            return output, None # 1, nword, track, length

class D_noise(nn.Module):

    def __init__(self, opt):
        super(D_noise, self).__init__()
        self.transformer = TransformerBlock_RGA(opt.ntrack, opt.nword)

    def forward(self, x, mems=None, draw_concat=False):
        x = self.transformer(x, draw_concat)
        return x, None

