import SMGAN.functions as functions
import SMGAN.model as model
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


lib = Lang("song")

def trainWOGAN(opt, Gs, Zs, reals, NoiseAmp):
    print("************** start training ****************")
    in_s = 0#
    num_scale = 0
    nfc_prev = 0#

    if opt.input_dir == 'array':
        real_ = functions.load_phrase_from_npy(opt)
    if opt.input_dir == 'midi':
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt)
    if opt.input_dir == 'pianoroll':
        real_ = functions.load_phrase_from_npz(opt)#原 5
    if opt.input_dir == 'JSB-Chorales-dataset':
        real_ = functions.load_phrase_from_pickle(opt)

    print(">> Input real_ shape = ", real_.shape)

    lib.addSong(real_) # for generating lib
    print(">> Total words = ", lib.n_words)
    opt.nword = lib.n_words
    reals = functions.get_reals(real_, reals, 16, [4,8,16])
    reals_num = list(map(lambda x:lib.song2num(x), reals))
    reals_num = list(map(lambda x: functions.np2torch(x), reals_num)) # track, bar, time

    reals = reals_num


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
        print(">> current real_scale shape is : ", real_scale.shape)
        merged = save_image('%s/real_scale.png' % (opt.outp), real_scale, (1, 1))
        save_midi('%s/real_scale.mid' % (opt.outp), real_scale, opt)
        # wandb.log({
        #     "real [%d]"% num_scale: wandb.Image(merged)},
        #     commit=False
        # )

        #初始化D,G模型       打印网络结构
        G_curr, D_curr= init_models(opt)
        if (nfc_prev == opt.nfc):#使num_scale-1从0开始  加载上一阶段训练好的模型
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out, num_scale-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out, num_scale-1)))

        G_curr = train_single_scale(D_curr,G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)#训练该阶段模型并保存成netG.pth, netD.pth

        G_curr = functions.reset_grads(G_curr,False)#????????
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)

        torch.save(Gs, '%s/Gs.pth' % (opt.out))

        num_scale += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr#把上一阶段数据清空
    return


def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    real = reals[len(Gs)]#len(Gs)=0  最小尺度的真值 3维   （track, bar, h）
    shape = real.shape
    real = real.reshape(1, shape[0], shape[1]*shape[2])#(1, track, bar*h）

    lowest_real = reals[0]
    lowest_shape = lowest_real.shape
    lowest_real = lowest_real.reshape(1, lowest_shape[0], lowest_shape[1]*lowest_shape[2])

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[3000],gamma=opt.gamma)

    errD2plot = []#损失
    errG2plot = []

    D_real2plot = []#D_x
    D_fake2plot = []#D_G_z
    z_opt2plot = []#rec_loss

    print('********************Training start!********************')
    dataset = batchify(real, int(4*(2**(len(Gs))))) # N, 1, track, length
    lowest_dataset = batchify(lowest_real, 4) # N, 1, track, length
    print(">>> the {}th stage, epoch is {}".format(
        len(Gs),
        (len(Gs)+1)*opt.niter
    ))
    for epoch in tqdm(range(opt.niter * (len(Gs) + 1))):#一阶段2000个epoch
        assert (dataset.shape[3] - int(4*(2**len(Gs))))/(2**len(Gs)) == int((dataset.shape[3] - int(4*(2**len(Gs))))/(2**len(Gs)))
        for i in range( int((dataset.shape[3] - int(4*(2**len(Gs))))/(2**len(Gs))) ):
            _, tgt = get_batch(dataset, i, int(4*(2**(len(Gs))))) # 1, track, length
            lowest_input, _ = get_batch(lowest_dataset, i, 4)

            src = draw_concat(Gs, lowest_input, opt)

            netG.zero_grad()
            # print("drawcat shape: ", src.shape)
            output, _ = netG(src, mode="nword") # 1, nwork, track, length
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, tgt.long())
            loss.backward()
            optimizerG.step()#网络参数更新

        # if epoch % 10 == 0 or epoch == (opt.niter-1):
        #     song = output.detach().argmax(dim=1) # [1, track, length]
        #     song = output.reshape([opt.ntrack, opt.nbar, -1])
        #     song = lib.num2song(functions.convert_image_np(output.detach()))[None, ]
        #     run_sampler(opt, song, epoch, postfix='666')

            wandb.log({
                "loss [%d]"% len(Gs): loss.detach()}
            )

        schedulerG.step()

    functions.save_networks(netG, netD, opt)
    return netG


def draw_concat(Gs, in_s, opt):
    G_z = in_s#第一阶段G_z=in_s(图片大小)
    if len(Gs) > 0:##其他阶段
        for G in Gs:
            ########################################！！！！！！！！！！第一阶段噪声每track相同
            G_z, _ = G(G_z, mode="top1")
            cur_scale = 2
            G_z = word_upsample(G_z, cur_scale)
    return G_z

#初始化模型
def init_models(opt):
    print(">> Input model type:", opt.model_type)
    if opt.model_type == "transformer":
        netG = model.G_transform(opt).to(opt.device)
        netG.apply(model.weights_init)
        if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
            netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
        # print(netG)#打印网络结构

        netD = model.D_transform(opt).to(opt.device)
        netD.apply(model.weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        # print(netD)#打印网络结构
    elif opt.model_type == "transformerXL":

        netG = model.G_transformXL(opt).to(opt.device)
        netG.apply(model.weights_init)
        if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
            netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
        # print(netG)#打印网络结构

        netD = model.D_transformXL(opt).to(opt.device)
        netD.apply(model.weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        # print(netD)#打印网络结构
    elif opt.model_type == "conv":
        netG = model.GeneratorConcatSkip2CleanAdd_init(opt).to(opt.device)
        netG.apply(model.weights_init)
        if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
            netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
        # print(netG)#打印网络结构

        netD = model.WDiscriminator_init(opt).to(opt.device)
        netD = model.D_transform(opt).to(opt.device)
        netD.apply(model.weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        # print(netD)#打印网络结构
    else:
        print("not select model type in args! maybe transformer, transformerXL, conv")
        exit(0)

    return netG, netD


# def init_models(opt):
#     netG_t = model.Generator_temp(opt)
#     netG_t = apply(model.weights_init)
#     if opt.netG_t != '':#若训练过程中断, 再次训练可接上次
#         netG.load_state_dict(torch.load(opt.netG_t))#加载预训练模型
#     print(netG_t)#打印网络结构

#     netG_b = model.Generator_bar(opt)
#     netG_b = apply(model.weights_init)
#     if opt.netG_b != '':#若训练过程中断, 再次训练可接上次
#         netG.load_state_dict(torch.load(opt.netG_b))#加载预训练模型
#     print(netG_b)#打印网络结构

#     netD = model.Discriminator(opt)
#     netD = apply(model.weights_init)
#     if opt.netD != '':
#         netD.load_state_dict(torch.load(opt.netD))
#     print(netD)#打印网络结构

#     return netG_t, netG_b, netD
