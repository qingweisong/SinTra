import SMGAN.functions as functions
import SMGAN.model as model
import SMGAN.model_GAN as gan
from SMGAN.in_out import *
from SMGAN.image_io import *
from SMGAN.implement import *
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from SMGAN.utils import Lang
import SMGAN.functions as functions
from SMGAN.xl import MemTransformerLM


wandb.init(
    project="xl",
    config = {
    }
)

lib = Lang("song")

def trainXL(opt, Gs, reals):
    print("************** start training ****************")
    in_s = 0#
    num_scale = 0
    nfc_prev = 0#

    if opt.input_dir == 'array':
        real_ = functions.load_phrase_from_npy(opt)
    if opt.input_dir == 'midi':
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt, all=True)
    if opt.input_dir == 'pianoroll':
        real_ = functions.load_phrase_from_npz(opt)#原 5
    if opt.input_dir == 'JSB-Chorales-dataset':
        real_ = functions.load_phrase_from_pickle(opt, all=True)

    print("Input real_ shape = ", real_.shape)

    lib.addSong(real_) # for generating lib
    print("Total words = ", lib.n_words)
    opt.nword = lib.n_words
    reals = functions.get_reals(real_, reals, 16, [4,8,16])
    reals_num = list(map(lambda x:lib.song2num(x), reals))
    reals_num = list(map(lambda x: functions.np2torch(x), reals_num)) # track, bar, time

    reals = reals_num

    print(reals[0].shape)
    opt.nbar = reals[0].shape[1]

    memGs = [tuple() for _ in range(len(reals))]
    while num_scale < len(reals): #opt.stop_scale + 1:#5
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(num_scale / 4)), 128)#32 (0-3)  64 (4-7) 128 (8-无穷大阶段)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(num_scale / 4)), 128)

        opt.out = functions.generate_dir2save(opt)
        opt.outp = '%s/%d' % (opt.out, num_scale)
        try:
            os.makedirs(opt.outp)
        except OSError:#上一句未执行成功就pass
            pass
        
        print("************* Save real_scale **************")
        real_scale = lib.num2song(functions.convert_image_np(reals[num_scale]))[None, ] # to np
        print("current real_scale shape is : ", end ="")
        print(real_scale.shape)
        merged = save_image('%s/real_scale.png' % (opt.outp), real_scale, (1, 1))
        save_midi('%s/real_scale.mid' % (opt.outp), real_scale, opt)
        # wandb.log({
        #     "real [%d]"% num_scale: wandb.Image(merged)},
        #     commit=False
        # )

        #初始化D,G模型       打印网络结构
        G_curr = init_models(opt, (2**num_scale)*4)
        if (nfc_prev == opt.nfc):#使num_scale-1从0开始  加载上一阶段训练好的模型
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out, num_scale-1)))

        G_curr = train_single_scale(G_curr, reals, Gs, in_s, opt)#训练该阶段模型并保存成netG.pth, netD.pth

        G_curr = functions.reset_grads(G_curr,False)#????????
        G_curr.eval()

        Gs.append(G_curr)

        torch.save(Gs, '%s/Gs.pth' % (opt.out))

        num_scale += 1
        nfc_prev = opt.nfc
        del G_curr#把上一阶段数据清空
    return


def train_single_scale(netG, reals, Gs, in_s, opt):
    real = reals[len(Gs)]#len(Gs)=0  最小尺度的真值 3维   （track, bar, h）
    shape = real.shape
    real = real.reshape(1, shape[0], shape[1]*shape[2])#(1, track, bar*h）

    lowest_real = reals[0]
    lowest_shape = lowest_real.shape
    lowest_real = lowest_real.reshape(1, lowest_shape[0], lowest_shape[1]*lowest_shape[2])

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []#损失
    errG2plot = []

    D_real2plot = []#D_x
    D_fake2plot = []#D_G_z
    z_opt2plot = []#rec_loss

    print('********************Training start!********************')
    dataset = batchify(real, int(4*(2**(len(Gs)))))
    lowest_dataset = batchify(lowest_real, 4)

    for epoch in tqdm(range(opt.niter)):#一阶段2000个epoch
        concat_mems = [tuple() for _ in range(len(Gs))]
        memG = tuple()
        for i in range(len(dataset) - 1):
            _, tgt = get_batch(dataset, i) # 1, track, length
            lowest_input, _ = get_batch(lowest_dataset, i)
            src = draw_concat(Gs, lowest_input, concat_mems, opt)

            netG.zero_grad()
            output, memG = netG(src, *memG, mode="loss") # 1, nwork, track, length
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tgt.long())
            loss.backward()
            optimizerG.step()#网络参数更新

            wandb.log({
                "loss [%d]"% len(Gs): loss.detach()}
            )

        schedulerG.step()

    functions.save_networks(netG, None, None, opt)
    return netG


def draw_concat(Gs, in_s, memGs, opt):
    G_z = in_s#第一阶段G_z=in_s(图片大小)
    if len(Gs) > 0:##其他阶段
        for i, G in enumerate(Gs):
            ########################################！！！！！！！！！！第一阶段噪声每track相同
            G_z, new_mem = G(G_z, *(memGs[i]), mode="top1")
            cur_scale = 2
            G_z = word_upsample(G_z, cur_scale, opt)
            memGs[i] = new_mem
    return G_z

#初始化模型
def init_models(opt, length):
    if opt.model_type == "msxl":

        netG = MemTransformerLM(
            n_token=lib.n_words,
            n_layer=6,
            n_track=opt.ntrack,
            n_head=4,
            d_model=256,
            d_head=32,
            d_inner=1024,
            dropout=0.1,
            dropatt=0.0,

            tgt_len=length,
            mem_len=length,
            ext_len=0
        ).to(opt.device)
        netG.apply(weights_init)

    else:
        print("not select model type in args! maybe transformer, transformerXL, conv")
        exit(0)

    return netG

def init_weight(weight):
    # if args.init == 'uniform':
    #     nn.init.uniform_(weight, -args.init_range, args.init_range)
    # elif args.init == 'normal':
    nn.init.normal_(weight, 0.0, 0.02)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)