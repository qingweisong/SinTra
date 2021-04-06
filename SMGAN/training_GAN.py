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

wandb.init(
    project="single_musegan_word_gan",
    config = {
    }
)

lib = Lang("song")

def trainWithGAN(opt, Gs, Ds, Zs, reals, NoiseAmp):
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
        G_curr, D_curr= init_models(opt)
        if (nfc_prev == opt.nfc):#使num_scale-1从0开始  加载上一阶段训练好的模型
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out, num_scale-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out, num_scale-1)))

        z_curr, in_s, G_curr = train_single_scale(D_curr,G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)#训练该阶段模型并保存成netG.pth, netD.pth

        G_curr = functions.reset_grads(G_curr,False)#????????
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)#噪声幅值

        torch.save(Zs, '%s/Zs.pth' % (opt.out))
        torch.save(Gs, '%s/Gs.pth' % (opt.out))
        torch.save(reals, '%s/reals.pth' % (opt.out))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out))

        num_scale += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr#把上一阶段数据清空
    return


def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    real = reals[len(Gs)]#len(Gs)=0  最小尺度的真值 3维   （track, bar, h）
    shape = real.shape
    real = real.reshape(1, shape[0], shape[1]*shape[2]).to("cuda") #(1, track, bar*h）

    lowest_real = reals[0]
    lowest_shape = lowest_real.shape
    lowest_real = lowest_real.reshape(1, lowest_shape[0], lowest_shape[1]*lowest_shape[2])

    opt.nzx = real.shape[1] # track
    opt.nzy = real.shape[2] # length

    fixed_z = functions.my_generate_noise([1, opt.nzx, opt.nzy, opt.noise_ninp], device = opt.device)#大小为(1, track, 4*h, w)的噪声矩阵
    z_opt = torch.full(fixed_z.shape, 0, device = opt.device)#全0矩阵  (b, c, 4*h, w)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
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

        if (Gs == []):#只有第一阶段有z_opt   生成重构图的噪声
            ########################################！！！！！！！！！！第一阶段噪声每track相同
            z_opt = functions.my_generate_noise([1, opt.nzx, opt.nzy, opt.noise_ninp], device=opt.device)#(1,1,4*h)
            noise_ = functions.my_generate_noise([1,opt.nzx, opt.nzy, opt.noise_ninp], device=opt.device)#(1,1,4*h)
        else:#其他阶段
            noise_ = functions.my_generate_noise([1, opt.nzx, opt.nzy, opt.noise_ninp], device=opt.device)


        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        memsD = tuple()
        memsG = tuple()
        for j in range(opt.Dsteps):
            netD.zero_grad()
            output, memsD = netD(real, None) # .to(opt.device)#real 4dim
            criterion = nn.CrossEntropyLoss()
            errD_real = criterion(output, real.long())
            # errD_real = -errD_real
            errD_real.backward(retain_graph = True)
            D_x = -errD_real.item()

            if (j==0) & (epoch == 0):#第一个epoch第一次训练D
                if (Gs == []):#第一阶段 prev指上一阶段的输出
                    prev = torch.full([1, opt.nzx, opt.nzy, opt.noise_ninp], 0, device=opt.device)
                    in_s = prev
                    z_prev = torch.full([1, opt.nzx, opt.nzy, opt.noise_ninp], 0, device=opt.device)
                    opt.noise_amp = 1
                else:#其他阶段
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',opt)#prev指上一阶段的输出
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',opt)#z_prev指上一阶段的重构图
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(z_prev.to("cuda"), real.long()) #重构数据和原始数据对应点误差的平方和的均值再开根
                    opt.noise_amp = opt.noise_amp_init*loss #addative noise cont weight * RMSELoss
            else:#非第一个epoch第一次训练D
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',opt)#假图


            if (Gs == []):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_ + prev.to("cuda") #噪*+ fake
            
            fake, _ = netG(noise.detach(), None, mode="top1")
            output, _ = netD(fake.detach(), None)
            criterion = nn.CrossEntropyLoss()
            errD_fake = criterion(output, real.long())
            errD_fake = -errD_fake
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            # gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)#具有梯度惩罚的WGAN（WGAN with gradient penalty）gradient penelty weight=0.1
            # gradient_penalty.backward()

            errD = errD_real + errD_fake #+ gradient_penalty
            optimizerD.step()#网络参数更新

        errD2plot.append(errD.detach())
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output, _ = netD(fake, None)
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if opt.alpha!= 0:#reconstruction loss weight=10
                criterion = nn.CrossEntropyLoss() # out is [N, class, d1] tgt is [N, d1]
                z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec_feature',opt)#z_prev指上一阶段的重构图
                print(z_opt.shape)
                print(z_prev.shape)
                Z_opt = opt.noise_amp*z_opt+z_prev.to("cuda") #该阶段重构输入
                print("======== in G update")
                print(netG(Z_opt.detach(), None)[0].shape)
                print(real.shape)
                rec_loss = opt.alpha*criterion(netG(Z_opt.detach(), None)[0], real.long())
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss.detach())
        z_opt2plot.append(rec_loss)

        ########################

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, z_opt, opt)
    print("<<<<<<<<<<<<<<<<<<<<<<< single train done!")
    return z_opt, in_s, netG


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, opt):
    G_z = in_s #第一阶段G_z=in_s(图片大小)
    end = len(Gs)
    print("========= in draw concat")
    print(len(Zs))
    if len(Gs) > 0:##其他阶段
        if mode == 'rand':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                ########################################！！！！！！！！！！第一阶段噪声每track相同
                if count == 0:
                    z = functions.my_generate_noise([1, 1, Z_opt.shape[2], Z_opt.shape[3]], device=opt.device)
                    z = z.expand(1, Z_opt.shape[1], z.shape[2], z.shape[3])
                else:
                    z = functions.my_generate_noise(Z_opt.shape , device=opt.device)
                z_in = noise_amp*z+G_z.to("cuda")
                G_z, _ = G(z_in.detach(), mode="feature")

                cur_scale = 2
                G_z = word_upsample_feature(G_z, cur_scale, opt)
                count += 1

        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == end - 1:
                    z_in = noise_amp*Z_opt+G_z
                    G_z, _ = G(z_in.detach(), mode="loss")
                    cur_scale = 2
                    G_z = word_upsample_nword(G_z, cur_scale, opt)
                else:
                    z_in = noise_amp*Z_opt+G_z
                    G_z, _ = G(z_in.detach(), mode="feature")
                    cur_scale = 2
                    G_z = word_upsample_feature(G_z, cur_scale, opt)
                count += 1

        if mode == 'rec_feature':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                z_in = noise_amp*Z_opt+G_z
                G_z, _ = G(z_in.detach(), mode="feature")
                cur_scale = 2
                G_z = word_upsample_feature(G_z, cur_scale, opt)
                count += 1
    return G_z

#初始化模型
def init_models(opt):
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

    elif opt.model_type == "gan":
        netG = gan.G_noise(opt).to(opt.device)
        netG.apply(model.weights_init)
        if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
            netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
        # print(netG)#打印网络结构

        netD = gan.D_noise(opt).to(opt.device)
        netD.apply(model.weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        # print(netD)#打印网络结构

    elif opt.model_type == "rga":
        netG = model.G_transformRGA(opt).to(opt.device)
        netG.apply(model.weights_init)
        if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
            netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
        # print(netG)#打印网络结构

        netD = model.D_transformRGA(opt).to(opt.device)
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