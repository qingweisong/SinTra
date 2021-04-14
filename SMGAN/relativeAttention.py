import SMGAN.utils as utils

import math as m
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = utils.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, **kwargs):
        attn_out, w = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)
        self.rga = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, mask=None, lookup_mask=None, w_out=False, **kwargs):

        attn_out, aw1 = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        if encode_out is None:
            attn_out2, aw2 = self.rga2([out1, out1, out1], mask=mask)
        else:
            attn_out2, aw2 = self.rga2([out1, encode_out, encode_out], mask=mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = F.relu(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2+ffn_out)

        if w_out:
            return out, aw1, aw2
        else:
            return out


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask)
            weights.append(w)
        return x, weights # (batch_size, input_seq_len, d_model)



# ==========================================================
class TransformerBlock_RGA(torch.nn.Module):

    def __init__(
            self,
            track,
            nword,
            ninp = 64,
            nhead = 8,
            nhid = 512,
            nlayers = 6,
            dropout=0.1,
            max_len=2048
    ):
        super(TransformerBlock_RGA, self).__init__()
        self.model_type = 'Transformer_RGA'
        self.src_mask = None
        self.track = track
        self.ninp = ninp
        self.max_seq = max_len
        self.nlayers = nlayers

        self.pos_encoding = DynamicPositionEmbedding(ninp, max_seq=max_len)

        # handle track dim
        self.track_conv1 = nn.Conv2d(track, 16, 1)
        self.track_conv2 = nn.Conv2d(16, 1, 1)
        self.track_deconv1 = nn.Conv2d(1, 16, 1)
        self.track_deconv2 = nn.Conv2d(16, track, 1)

        # embeding
        # 1st: pitch -> 64 -> 32 -> 1
        # 2rd: track -> feature
        self.embeding = nn.Embedding(nword, ninp)

        # transformer rga
        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(ninp, dropout, h=ninp // 16, additional=False, max_seq=max_len)
             for _ in range(nlayers)])
        self.dropout = torch.nn.Dropout(dropout)

        # deocder
        self.decoder = nn.Linear(ninp, nword)
        self.decoders = nn.ModuleList([nn.Linear(ninp, nword) for i in range(track)])

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

    def topP_sampling(self, x, p=0.6): # 1, track, length, nword
        xp = F.softmax(x, dim=-1) # 1, track, length, nword
        topP, indices = torch.sort(x, dim=-1, descending=True)
        cumsum = torch.cumsum(topP, dim=-1)
        mask = torch.where(cumsum < p, topP, torch.ones_like(topP)*1000)
        minP, indices = torch.min(mask, dim=-1, keepdim=True)
        valid_p = torch.where(xp<minP, torch.ones_like(x)*(1e-10), xp)
        sample = torch.distributions.Categorical(valid_p).sample()
        return sample # 1, track, lengtha

    def forward(self, img, mode, p=0.6):

        img = img.long().cuda() # N, track, length

        tmp = img[:, 0, :]
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(tmp.shape[1], tmp, tmp, -10)

        src = self.embeding(img) # N, track, length, feature

        src = self.track_conv1(src) # N, 16, length, feature
        src = self.track_conv2(src) # N, 1, length, feature
        src = src[:, 0, :, :] # N, length, feature

        src = src * math.sqrt(self.ninp)

        weights = []
        x = src
        for i in range(self.nlayers):
            x, w = self.enc_layers[i](x, look_ahead_mask)
            weights.append(w)

        output = x[:, None, :, :] # 1, 1, length, feature
        output = self.track_deconv1(output) # 1, 16, length, feature
        output = self.track_deconv2(output) # 1, track, length, feature

        outputs = []
        for i in range(self.track):
            a_track = self.decoders[i](output[:, i, :, :]) # 1, length, nword
            a_track = a_track[:, None, :, :] # 1, 1, length, nword
            outputs.append(a_track)

        output = torch.cat(outputs, dim=1) # 1, track, length, nword

        if mode == "top1":
            top1 = output # 1, track, length, nword
            top1 = top1.argmax(dim=3) # 1, track, length
            return top1 
        elif mode == "topP":
            sample = self.topP_sampling(output, p) # 1, track, length
            return sample
        elif mode == "nword":
            output = output.permute(0, 3, 1, 2) # 1, nword, track, length

            return output # 1, nword, track, length
