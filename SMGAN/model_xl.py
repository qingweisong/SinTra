import torch
import torch.nn as nn
import torch.nn.functional as F
from SMGAN.xl import *


class Preprocess(nn.Module):

    def __init__(self, track, ninp):
        super(Preprocess, self).__init__()

        self.en_pitch1 = nn.Linear(128, ninp)
        self.en_track = nn.Linear(track, 1)
        self.en_pitch2 = nn.Conv2d(ninp, ninp, 1) # track 2 feature

    def forward(self, ppr): # 1, track, time, pitch
        src = self.en_pitch1(ppr)
        src = src.permute(0, 3, 2, 1).contiguous()
        src = self.en_track(src)
        src = self.en_pitch2(src)
        return src          # 1, pitch2feature, time, trackTo1


class Postprocess(nn.Module):

    def __init__(self, track, ninp):
        super(Postprocess, self).__init__()

        self.de_pitch = nn.Conv2d(ninp, 128, 1)
        self.de_track = nn.Linear(1, track)

    def forward(self, mid): # 1, ninp, time, 1
        src = self.de_pitch(mid) # 1, 128, time, 1
        src = self.de_track(src) # 1, 128, time, track
        src = src.permute(0, 3, 2, 1).contiguous() # 1, track, time , 128
        return src


class TransformerXL(nn.Module):
    
    def __init__(self, track, n_layer=6, n_head=4, d_model=256, d_head=32, d_inner=1024,
                 dropout=0.1, dropatt=0.0, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1):
        
        super(TransformerXL, self).__init__()

        # some variablle
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        # embeding 
        self.embeding = Preprocess(track, d_model)
        self.decoder = Postprocess(track, d_model)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        # transformer
        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        # elif attn_type == 1: # learnable embeddings
        #     for i in range(n_layer):
        #         self.layers.append(
        #             RelLearnableDecoderLayer(
        #                 n_head, d_model, d_head, d_inner, dropout,
        #                 tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
        #                 dropatt=dropatt, pre_lnorm=pre_lnorm)
        #         )
        # elif attn_type in [2, 3]: # absolute embeddings
        #     for i in range(n_layer):
        #         self.layers.append(
        #             DecoderLayer(
        #                 n_head, d_model, d_head, d_inner, dropout,
        #                 dropatt=dropatt, pre_lnorm=pre_lnorm)
        #         )


        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        _, _, qlen, _ = dec_inp.size() # 1, 1, 32, 128: bsz, track, time, pitch

        word_emb = self.embeding(dec_inp) # 1, 256, 32, 1: bsz, pitch2feature, time, trakcTo1
        word_emb = word_emb[:, :, :, 0].permute(2, 0, 1) # 32, 1, 256

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]
        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out) # 32, 1, 256: time, 1, feature

        core_out = self.decoder(core_out.permute(2, 0, 1).contiguous()[None, :, :, :])

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, dec_inp, mems):
        core_out, mems = self._forward(dec_inp, mems)
        return core_out, mems


class SinMuse(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass