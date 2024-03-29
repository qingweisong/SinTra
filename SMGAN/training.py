# import SMGAN.functions as functions
# import SMGAN.model as model
# from SMGAN.in_out import *
# from SMGAN.image_io import *
# from SMGAN.implement import *
# import os
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data
# import math
# import matplotlib.pyplot as plt
# import wandb
# from tqdm import tqdm
# from SMGAN.utils import Lang
# import SMGAN.functions as functions

# wandb.init(
#     project="single-musegan",
#     config = {
#     }
# )

# lib = Lang("song")

# def trainWOGAN(opt, Gs, Zs, reals, NoiseAmp):
#     print("************** start training ****************")
#     in_s = 0#
#     num_scale = 0
#     nfc_prev = 0#

#     if opt.input_dir == 'array':
#         real_ = functions.load_phrase_from_npy(opt)
#     if opt.input_dir == 'midi':
#         real_ = midi2np(opt)
#         real_ = midiArrayReshape(real_, opt)
#     if opt.input_dir == 'pianoroll':
#         real_ = functions.load_phrase_from_npz(opt)#原 5
#     if opt.input_dir == 'JSB-Chorales-dataset':
#         real_ = functions.load_phrase_from_pickle(opt)

#     print("Input real_ shape = ", real_.shape)

#     lib.addSong(real_) # for generating lib
#     print("Total words = ", lib.n_words)
#     reals = functions.get_reals(real_, reals, 16, [4,8,16])
#     reals_num = list(map(lambda x:lib.song2num(x), reals))
#     reals_num = list(map(lambda x: functions.np2torch(x), reals_num)) # track, bar, t

#     reals = reals_num

#     # print(reals[0].shape)
#     opt.nbar = reals[0].shape[1]

#     while num_scale < len(reals): #opt.stop_scale + 1:#5
#         opt.nfc = min(opt.nfc_init * pow(2, math.floor(num_scale / 4)), 128)#32 (0-3)  64 (4-7) 128 (8-无穷大阶段)
#         opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(num_scale / 4)), 128)

#         opt.out = functions.generate_dir2save(opt)
#         opt.outp = '%s/%d' % (opt.out, num_scale)
#         try:
#             os.makedirs(opt.outp)
#         except OSError:#上一句未执行成功就pass
#             pass
        
#         print("************* Save real_scale **************")
#         real_scale = lib.num2song(functions.convert_image_np(reals[num_scale]))[None, ] # to np
#         print("current real_scale shape is : ",real_scale.shape)
#         merged = save_image('%s/real_scale.png' % (opt.outp), real_scale, (1, 1))
#         save_midi('%s/real_scale.mid' % (opt.outp), real_scale, opt)
#         wandb.log({
#             "real [%d]"% num_scale: wandb.Image(merged)},
#             commit=False
#         )

#         #初始化D,G模型       打印网络结构
#         G_curr, D_curr= init_models(opt)
#         if (nfc_prev == opt.nfc):#使num_scale-1从0开始  加载上一阶段训练好的模型
#             G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out, num_scale-1)))
#             D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out, num_scale-1)))

#         z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)#训练该阶段模型并保存成netG.pth, netD.pth

#         G_curr = functions.reset_grads(G_curr,False)#????????
#         G_curr.eval()
#         D_curr = functions.reset_grads(D_curr,False)
#         D_curr.eval()

#         Gs.append(G_curr)
#         Zs.append(z_curr)
#         NoiseAmp.append(opt.noise_amp)#噪声幅值

#         torch.save(Zs, '%s/Zs.pth' % (opt.out))
#         torch.save(Gs, '%s/Gs.pth' % (opt.out))
#         torch.save(reals, '%s/reals.pth' % (opt.out))
#         torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out))

#         num_scale += 1
#         nfc_prev = opt.nfc
#         del D_curr, G_curr#把上一阶段数据清空
#     return



#     return

# def train(opt, Gs, Zs, reals, NoiseAmp):
#     print("************** start training ****************")
#     in_s = 0#
#     num_scale = 0
#     nfc_prev = 0#

#     if opt.input_dir == 'array':
#         real_ = functions.load_phrase_from_npy(opt)
#     if opt.input_dir == 'midi':
#         real_ = midi2np(opt)
#         real_ = midiArrayReshape(real_, opt)
#     if opt.input_dir == 'pianoroll':
#         real_ = functions.load_phrase_from_npz(opt)#原 5
#     if opt.input_dir == 'JSB-Chorales-dataset':
#         real_ = functions.load_phrase_from_pickle(opt, all=True)

#     print("Input real_ shape = ", real_.shape)

#     # real = functions.imresize_in(real_, opt.scale1)#max 5维(np类型)
#     # reals = functions.creat_reals_pyramid(real, reals, opt)#一组不同尺寸的phrase真值(torch类型) cuda上

#     # handle data
#     lib.addSong(real_) # for generating lib
#     print("Total words = ", lib.n_words)
#     reals = functions.get_reals(real_, reals, 16, [4,8,16])
#     # print("==================")
#     # print(lib.pitch2tuple(reals[0]))
#     # print("==================")
#     # print(lib.pitch2tuple(real_))
#     reals_num = list(map(lambda x:lib.song2num(x), reals))
#     reals_num = list(map(lambda x: functions.np2torch(x), reals_num)) # track, bar, t

#     reals = reals_num

#     # print(reals[0].shape)
#     opt.nbar = reals[0].shape[1]

#     # scale = [2, 4]
#     # srcs = []
#     # tgts = []
#     # for i in range(len(scale)):
#     #     src, tgt = functions.batchify(real_, scale[i])
#     #     srcs.append(src)
#     #     reals.append(src)
#     #     tgts.append(tgt)


#     while num_scale < len(reals): #opt.stop_scale + 1:#5
#         opt.nfc = min(opt.nfc_init * pow(2, math.floor(num_scale / 4)), 128)#32 (0-3)  64 (4-7) 128 (8-无穷大阶段)
#         opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(num_scale / 4)), 128)

#         opt.out = functions.generate_dir2save(opt)
#         opt.outp = '%s/%d' % (opt.out, num_scale)
#         try:
#             os.makedirs(opt.outp)
#         except OSError:#上一句未执行成功就pass
#             pass
        
#         print("************* Save real_scale **************")
#         real_scale = lib.num2song(functions.convert_image_np(reals[num_scale]))[None, ] # to np
#         print("current real_scale shape is : ", end ="")
#         print(real_scale.shape)
#         merged = save_image('%s/real_scale.png' % (opt.outp), real_scale, (1, 1))
#         save_midi('%s/real_scale.mid' % (opt.outp), real_scale, opt)
#         wandb.log({
#             "real [%d]"% num_scale: wandb.Image(merged)},
#             commit=False
#         )

#         #初始化D,G模型       打印网络结构
#         G_curr, D_curr= init_models(opt)
#         if (nfc_prev == opt.nfc):#使num_scale-1从0开始  加载上一阶段训练好的模型
#             G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out, num_scale-1)))
#             D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out, num_scale-1)))

#         z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)#训练该阶段模型并保存成netG.pth, netD.pth

#         G_curr = functions.reset_grads(G_curr,False)#????????
#         G_curr.eval()
#         D_curr = functions.reset_grads(D_curr,False)
#         D_curr.eval()

#         Gs.append(G_curr)
#         Zs.append(z_curr)
#         NoiseAmp.append(opt.noise_amp)#噪声幅值

#         torch.save(Zs, '%s/Zs.pth' % (opt.out))
#         torch.save(Gs, '%s/Gs.pth' % (opt.out))
#         torch.save(reals, '%s/reals.pth' % (opt.out))
#         torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out))

#         num_scale += 1
#         nfc_prev = opt.nfc
#         del D_curr, G_curr#把上一阶段数据清空
#     return


# def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
#     real = reals[len(Gs)]#len(Gs)=0  最小尺度的真值 3维   （track, bar, h）
#     shape = real.shape
#     real = real.reshape(1, shape[0], shape[1]*shape[2])#(1, track, bar*h）


#     opt.nzx = real.shape[2]#bar * h
#     # opt.nzy = 1
#     pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)#5？？
#     pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)#5
#     m_noise = nn.ZeroPad2d(0)#上下左右pad5行
#     m_image = nn.ZeroPad2d(0)

#     fixed_z = functions.generate_noise([real.shape[1],opt.nzx], num_samp = opt.nsample, device = opt.device)#大小为(1, track, 4*h, w)的噪声矩阵
#     z_opt = torch.full(fixed_z.shape, 0, device = opt.device)#全0矩阵  (b, c, 4*h, w)
#     z_opt = m_noise(z_opt)#上下左右pad5行
    
#     # setup optimizer
#     optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
#     schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
#     schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

#     # ============================ step 5/5 训练 ============================
#     errD2plot = []#损失
#     errG2plot = []

#     D_real2plot = []#D_x
#     D_fake2plot = []#D_G_z
#     z_opt2plot = []#rec_loss

#     print('********************Training start!********************')
#     for epoch in tqdm(range(opt.niter)):#一阶段2000个epoch
#         if (Gs == []):#只有第一阶段有z_opt   生成重构图的噪声
#             ########################################！！！！！！！！！！第一阶段噪声每track相同
#             z_opt = functions.generate_noise([1,opt.nzx], device=opt.device)#(1,1,4*h,w)
#             z_opt = m_noise(z_opt.expand(1,opt.ntrack,opt.nzx))#(1,5,4*h+10,w+10)
#             noise_ = functions.generate_noise([1,opt.nzx], device=opt.device)#(1,1,4*h,w)
#             noise_ = m_noise(noise_.expand(1,opt.ntrack,opt.nzx))#(1,5,4*h+10,w+10)
#         else:#其他阶段
#             noise_ = functions.generate_noise([opt.ntrack,opt.nzx], num_samp = opt.nsample, device=opt.device)
#             noise_ = m_noise(noise_)

#         ############################
#         # (1) Update D network: maximize D(x) + D(G(z))
#         ###########################
#         memsD = tuple()
#         memsG = tuple()
#         for j in range(opt.Dsteps):
#             netD.zero_grad()
#             output, memsD = netD(real, None) # .to(opt.device)#real 4dim
#             errD_real = -output.mean()
#             errD_real.backward(retain_graph = True)
#             D_x = -errD_real.item()

#             if (j==0) & (epoch == 0):#第一个epoch第一次训练D
#                 if (Gs == []):#第一阶段 prev指上一阶段的输出
#                     prev = torch.full([1,opt.ntrack,opt.nzx], 0, device=opt.device)
#                     in_s = prev
#                     prev = m_image(prev)
#                     z_prev = torch.full([1,opt.ntrack,opt.nzx], 0, device=opt.device)
#                     z_prev = m_noise(z_prev)
#                     opt.noise_amp = 1
#                 else:#其他阶段
#                     prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)#prev指上一阶段的输出
#                     prev = m_image(prev)#pad后
#                     z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)#z_prev指上一阶段的重构图
#                     print(real.shape)
#                     print(z_prev.shape)
#                     criterion = nn.MSELoss()
#                     RMSE = torch.sqrt(criterion(real, z_prev))#重构数据和原始数据对应点误差的平方和的均值再开根
#                     opt.noise_amp = opt.noise_amp_init*RMSE#addative noise cont weight * RMSELoss
#                     print(opt.noise_amp_init)
#                     print(opt.noise_amp)
#                     z_prev = m_image(z_prev)
#             else:#非第一个epoch第一次训练D
#                 prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)#假图
#                 prev = m_image(prev)#pad后


#             if (Gs == []):
#                 noise = noise_
#             else:
#                 noise = opt.noise_amp*noise_ + prev#噪*+ fake
            
#             print("!!!!!!!", noise.shape)
#             print("!!!!!!!", prev.shape)
#             fake, _ = netG(noise.detach(),prev, None)
#             print("???????", fake.shape)
#             output, _ = netD(fake.detach(), None)
#             errD_fake = output.mean()
#             errD_fake.backward(retain_graph=True)
#             D_G_z = output.mean().item()

#             gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)#具有梯度惩罚的WGAN（WGAN with gradient penalty）gradient penelty weight=0.1
#             gradient_penalty.backward()

#             errD = errD_real + errD_fake + gradient_penalty
#             optimizerD.step()#网络参数更新

#         errD2plot.append(errD.detach())
#         D_real2plot.append(D_x)
#         D_fake2plot.append(D_G_z)

#         ############################
#         # (2) Update G network: maximize D(G(z))
#         ###########################

#         for j in range(opt.Gsteps):
#             netG.zero_grad()
#             output, _ = netD(fake, None)
#             errG = -output.mean()
#             errG.backward(retain_graph=True)
#             if opt.alpha!= 0:#reconstruction loss weight=10
#                 loss = nn.NLLLoss() # out is [N, class, d1] tgt is [N, d1]
#                 Z_opt = opt.noise_amp*z_opt+z_prev#该阶段重构输入
#                 rec_loss = opt.alpha*loss(netG(Z_opt.detach(),z_prev)[0],real)

#                 # loss = nn.MSELoss()#recloss
#                 # Z_opt = opt.noise_amp*z_opt+z_prev#该阶段重构输入
#                 # rec_loss = opt.alpha*loss(netG(Z_opt.detach(),z_prev)[0],real)

#                 rec_loss.backward(retain_graph=True)
#                 rec_loss = rec_loss.detach()
#             optimizerG.step()

#         errG2plot.append(errG.detach()+rec_loss.detach())
#         z_opt2plot.append(rec_loss)

#         #4tendor->5np
#         def cvt_5d(fake):
#             fake = fake.detach().cpu().numpy() # 1, track, bar*time
#             fake = fake[0, :, :]
#             fake = fake.reshape(fake.shape[0], opt.nbar, -1)
#             fake = lib.num2song(fake)[None, :, :, :, :]
#             return fake
#         fake = cvt_5d(fake)
#         fake = cvt_5d(fake)
#         fake = cvt_5d(fake)
#         # fake = dim_transformation_to_5(fake.detach(), opt).numpy()
#         rec  = netG(Z_opt.detach(), z_prev)[0].detach()
#         rec = cvt_5d(rec)
#         # rec = dim_transformation_to_5(rec, opt).numpy()

#         #bool类型的矩阵   round bernoulli
#         round = fake > 0.5
#         bernoulli = fake > torch.rand(fake.shape).numpy()
#         #denoise = functions.denoise_5D(round)


#         wandb.log({
#             "errD [%d]"%len(Gs): errD,
#             "errG [%d]"%len(Gs): errG,
#             "rec_loss [%d]"%len(Gs): rec_loss
#         })

#         if epoch % 100 == 0 or epoch == (opt.niter-1):
#             print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
#         # run sampler
#         if epoch % 500 == 0 or epoch == (opt.niter-1):
#             Fake = run_sampler(opt, fake, epoch, midi = 'False')
#             Round = run_sampler(opt, round, epoch, postfix='round')
#             #Denoise = run_sampler(opt, denoise, epoch, postfix='denoise')
#             Bernoulli = run_sampler(opt, bernoulli, epoch, postfix='bernoulli')
        
#             Rec = save_image('%s/G(z_opt).png' % (opt.outp), rec, (1, 1))
#             save_midi('%s/G(z_opt).mid' % (opt.outp), rec, opt)
#             torch.save(z_opt, '%s/z_opt.pth' % (opt.outp))

#         if epoch == (opt.niter-1):
#             wandb.log({
#                 "G(z) [%d]"%len(Gs): wandb.Image(Fake),
#                 "G(z_opt) [%d]"%len(Gs): wandb.Image(Rec),
#                 "Bernoulli [%d]"%len(Gs): wandb.Image(Bernoulli),
#                 "Round [%d]" % len(Gs): wandb.Image(Round)},
#                 #"Denoise [%d]"%len(Gs): wandb.Image(Denoise)},
#                 sync=False, commit=False
#             )

#         schedulerD.step()
#         schedulerG.step()

#     functions.save_networks(netG, netD, z_opt, opt)
#     return z_opt, in_s, netG


# def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
#     G_z = in_s#第一阶段G_z=in_s(图片大小)
#     if len(Gs) > 0:##其他阶段
#         if mode == 'rand':
#             count = 0
#             # pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
#             pad_noise = 0
#             for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
#                 ########################################！！！！！！！！！！第一阶段噪声每track相同
#                 if count == 0:
#                     z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
#                     z = z.expand(1, opt.ntrack, z.shape[2], z.shape[3])
#                 else:
#                     z = functions.generate_noise([opt.ntrack, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
#                 z = m_noise(z)
#                 G_z = G_z[:,:,0:real_curr.shape[2]*real_curr.shape[3],0:real_curr.shape[4]]
#                 G_z = m_image(G_z)
#                 z_in = noise_amp*z+G_z
#                 G_z, _ = G(z_in.detach(),G_z)
#                 cur_scale = real_next.shape[3] / (real_curr.shape[3]) 
#                 G_z = imresize(dim_transformation_to_5(G_z, opt),cur_scale,opt)#上采样到下一尺度大小
#                 #G_z = imresize(dim_transformation_to_5(G_z, opt),1/opt.scale_factor,opt, is_net = True)#上采样到下一尺度大小
#                 G_z = dim_transformation_to_4(G_z)[:,:,0:real_next.shape[2]*real_next.shape[3],0:real_next.shape[4]]

#                 count += 1
#         if mode == 'rec':
#             count = 0
#             for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
#                 G_z = G_z[:, :, 0:real_curr.shape[2]*real_curr.shape[3], 0:real_curr.shape[4]]
#                 G_z = m_image(G_z)
#                 z_in = noise_amp*Z_opt+G_z
#                 G_z, _ = G(z_in.detach(),G_z)


#                 cur_scale = real_next.shape[3] / (real_curr.shape[3]) 
#                 # print(real_next.shape)
#                 # print(real_curr.shape)
#                 # print(G_z.shape)
#                 # print(cur_scale)
#                 G_z = imresize(dim_transformation_to_5(G_z, opt),cur_scale,opt)
#                 #G_z = imresize(dim_transformation_to_5(G_z, opt),1/opt.scale_factor,opt, is_net = True)
#                 G_z = dim_transformation_to_4(G_z)[:,:,0:real_next.shape[2]*real_next.shape[3],0:real_next.shape[4]]
#                 #if count != (len(Gs)-1):
#                 #    G_z = m_image(G_z)
#                 count += 1
#     return G_z

# #初始化模型
# def init_models(opt):
#     if opt.model_type == "transformer":
#         netG = model.G_transform(opt).to(opt.device)
#         netG.apply(model.weights_init)
#         if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
#             netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
#         # print(netG)#打印网络结构

#         netD = model.D_transform(opt).to(opt.device)
#         netD.apply(model.weights_init)
#         if opt.netD != '':
#             netD.load_state_dict(torch.load(opt.netD))
#         # print(netD)#打印网络结构
#     elif opt.model_type == "transformerXL":

#         netG = model.G_transformXL(opt).to(opt.device)
#         netG.apply(model.weights_init)
#         if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
#             netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
#         # print(netG)#打印网络结构

#         netD = model.D_transformXL(opt).to(opt.device)
#         netD.apply(model.weights_init)
#         if opt.netD != '':
#             netD.load_state_dict(torch.load(opt.netD))
#         # print(netD)#打印网络结构
#     elif opt.model_type == "conv":
#         netG = model.GeneratorConcatSkip2CleanAdd_init(opt).to(opt.device)
#         netG.apply(model.weights_init)
#         if opt.netG != '':#若训练过程中断, 再次训练可接上次(一般不进入)
#             netG.load_state_dict(torch.load(opt.netG))#加载预训练模型
#         # print(netG)#打印网络结构

#         netD = model.WDiscriminator_init(opt).to(opt.device)
#         netD = model.D_transform(opt).to(opt.device)
#         netD.apply(model.weights_init)
#         if opt.netD != '':
#             netD.load_state_dict(torch.load(opt.netD))
#         # print(netD)#打印网络结构
#     else:
#         print("not select model type in args! maybe transformer, transformerXL, conv")
#         exit(0)

#     return netG, netD


# # def init_models(opt):
# #     netG_t = model.Generator_temp(opt)
# #     netG_t = apply(model.weights_init)
# #     if opt.netG_t != '':#若训练过程中断, 再次训练可接上次
# #         netG.load_state_dict(torch.load(opt.netG_t))#加载预训练模型
# #     print(netG_t)#打印网络结构

# #     netG_b = model.Generator_bar(opt)
# #     netG_b = apply(model.weights_init)
# #     if opt.netG_b != '':#若训练过程中断, 再次训练可接上次
# #         netG.load_state_dict(torch.load(opt.netG_b))#加载预训练模型
# #     print(netG_b)#打印网络结构

# #     netD = model.Discriminator(opt)
# #     netD = apply(model.weights_init)
# #     if opt.netD != '':
# #         netD.load_state_dict(torch.load(opt.netD))
# #     print(netD)#打印网络结构

# #     return netG_t, netG_b, netD
