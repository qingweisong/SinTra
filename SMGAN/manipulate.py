from __future__ import print_function
import SMGAN.functions
import SMGAN.model
import argparse
import os
import random
from SMGAN.functions import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SMGAN.training_noGAN import *
from config import get_arguments
from SMGAN.metrics import Metrics
import sys
import muspy
from tqdm import tqdm
import wandb

def generate_config(opt):
    config = {}

    config['scale_mask'] = list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]))
    config['tonal_matrix_coefficient'] = (1., 1., .5)
    config['tonal_distance_pairs'] = [(i, j) for i in range(opt.ntrack) if(i != 1)&(i!=4) for j in range(i+1, opt.ntrack) if(j != 1)&(j!=4)]#去掉重复s
    config['drum_filter']= np.tile([1., .1, 0., 0., 0., .1], 16)
    config['beat_resolution'] = 24    # ???

    config['track_names'] = opt.program_num
    config['metric_map'] = np.array([
        # indices of tracks for the metrics to compute
        [True] * opt.ntrack,  # empty bar rate
        [True] * opt.ntrack, # number of pitch used
        list(np.array(opt.is_drum) == False), # qualified note rate
        list(np.array(opt.is_drum) == False), # polyphonicity
        list(np.array(opt.is_drum) == False), # in scale rate
        opt.is_drum,          # in drum pattern rate
        list(np.array(opt.is_drum) == False)  # number of chroma used
    ], dtype=bool)

    return config

class Evaluation:
    def __init__(self, config):
        self.metric = Metrics(config)
        self.score_matrix_mean = []
        self.score_pair_matrix_mean = []

    def run_eval(self, target, postfix=None):
        target = target.transpose(0, 2, 3, 4, 1)
        binarized = target > 0
        if postfix is None:
            filename = "www"
        else:
            filename = "www" + '_' + postfix
        reshaped = binarized.reshape((-1,) + binarized.shape[2:])
        a, b = self.metric.eval(reshaped, mat_path=None)
        self.score_matrix_mean.append(a)
        self.score_pair_matrix_mean.append(b)
        return a, b
    
    def write_txt(self, mode="normal"):

        if mode == "real":
            info_dict = {
                'score_matrix_mean': self.score_matrix_mean,
                'score_pair_matrix_mean': self.score_pair_matrix_mean,
                }
            print('[*] Saving score matrices...')

            with open("./Metrics/" + mode  + ".txt", "w") as f:
                origin_stdout = sys.stdout
                sys.stdout = f

                print("========= This is in track")
                self.metric.print_metrics_mat(self.score_matrix_mean[0])

                print("\n")
                print("\n")
                print("\n")
                print("\n")

                print("========= This is track vs track")
                self.metric.print_metrics_pair(self.score_pair_matrix_mean[0])
                    
                print("\n")
                print("\n")
                print("\n")
                print("\n")
                sys.stdout = origin_stdout





        if mode != "real":
            # calculate average
            average_score_matrix_mean = 0
            for i in self.score_matrix_mean:
                average_score_matrix_mean += i
            average_score_matrix_mean /= (len(self.score_matrix_mean)*1.0)

            average_score_pair_matrix_mean = 0
            for i in self.score_pair_matrix_mean:
                average_score_pair_matrix_mean += i
            average_score_pair_matrix_mean /= (len(self.score_matrix_mean)*1.0)


            info_dict = {
                'score_matrix_mean': self.score_matrix_mean,
                'score_pair_matrix_mean': self.score_pair_matrix_mean,
                'average_score_matrix_mean': average_score_matrix_mean,
                'average_score_pair_matrix_mean': average_score_pair_matrix_mean
                }
            print('[*] Saving score matrices...')

            with open("./Metrics/" + mode  + ".txt", "w") as f:
                origin_stdout = sys.stdout
                sys.stdout = f
                print("========= This is average in track")
                self.metric.print_metrics_mat(average_score_matrix_mean)
                print("========= This is average track vs track")
                self.metric.print_metrics_pair(average_score_pair_matrix_mean)
            
                sys.stdout = origin_stdout



def SMGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=10):
    config = generate_config(opt)
    metric = Evaluation(config)
    denoise_metric = Evaluation(config)
    '''
    pitch_range
    n_pitches_used
    n_pitch_classes_used
    polyphony
    polyphony_rate
    pitch_in_scale_rate
    scale_consistency
    pitch_entropy
    pitch_class_entropy
    empty_beat_rate
    drum_in_pattern_rate_duple
    drum_in_pattern_rate_triple
    drum_pattern_consistency
    groove_consistency_64
    muspy.empty_measure_rate_64
    '''
    score_origin = []
    score_denoise = []
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
        in_s = dim_transformation_to_4(in_s)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        # pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        pad1 = 0
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in tqdm(range(0,num_samples,1)):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,opt.ntrack ,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.ntrack,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(dim_transformation_to_5(I_prev, opt), 2, opt)
                #I_prev = imresize(dim_transformation_to_5(I_prev, opt),1/opt.scale_factor, opt, is_net = True)
                I_prev = dim_transformation_to_4(I_prev)[:, :, 0:round(scale_v * reals[n].shape[2] * reals[n].shape[3]), 0:round(scale_h * reals[n].shape[4])]
                I_prev = m(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr, _ = G(z_in.detach(),I_prev)#tensor 4

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_phrase[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                fake = functions.dim_transformation_to_5(I_curr.detach(), opt).numpy()#np 5
                test_round = fake > 0.5
                # a, b = metric.run_eval(test_round.copy(), str(i))
                save_image('%s/%d.png' % (dir2save, i), test_round.copy(), (1,1))
                multitrack = save_midi('%s/%d.mid' % (dir2save, i), test_round.copy(), opt)
                # import ipdb; ipdb.set_trace()
                music = muspy.from_pypianoroll(multitrack)
                score_origin.append([
                    muspy.pitch_range(music),
                    muspy.n_pitches_used(music),
                    muspy.n_pitch_classes_used(music),
                    muspy.polyphony(music),
                    muspy.polyphony_rate(music, threshold=2),
                    muspy.pitch_in_scale_rate(music, root=1, mode='major'),
                    muspy.scale_consistency(music),
                    muspy.pitch_entropy(music),
                    muspy.pitch_class_entropy(music),
                    muspy.empty_beat_rate(music),
                    muspy.drum_in_pattern_rate(music, meter='duple'),
                    muspy.drum_pattern_consistency(music),
                    muspy.groove_consistency(music, measure_resolution=4096),
                    muspy.empty_measure_rate(music, measure_resolution=4096)
                ])
                
                denoise = functions.denoise_5D(test_round.copy(), opt)
                # a, b = denoise_metric.run_eval(denoise, str(i))
                save_image('%s/%d_denoise.png' % (dir2save, i), denoise.copy(), (1,1))
                multitrack = save_midi('%s/%d_denoise.mid' % (dir2save, i), denoise.copy(), opt)
                music = muspy.from_pypianoroll(multitrack)
                score_denoise.append([
                    muspy.pitch_range(music),
                    muspy.n_pitches_used(music),
                    muspy.n_pitch_classes_used(music),
                    muspy.polyphony(music),
                    muspy.polyphony_rate(music, threshold=2),
                    muspy.pitch_in_scale_rate(music, root=1, mode='major'),
                    muspy.scale_consistency(music),
                    muspy.pitch_entropy(music),
                    muspy.pitch_class_entropy(music),
                    muspy.empty_beat_rate(music),
                    muspy.drum_in_pattern_rate(music, meter='duple'),
                    muspy.drum_pattern_consistency(music),
                    muspy.groove_consistency(music, measure_resolution=4096),
                    muspy.empty_measure_rate(music, measure_resolution=4096)
                ])

                #metric.run_eval(test_round)
            images_cur.append(I_curr)
        n+=1
    # metric.write_txt()
    # denoise_metric.write_txt("denoise")
    return I_curr.detach()


def save_pic_midi(song, dir, index, opt, beat_resolution, wandb_enable):
    pic = save_image('%s/%d.png' % (dir, index), song.copy(), (1,1))
    multitrack = save_midi('%s/%d.mid' % (dir, index), song.copy(), opt, beat_resolution)

    if wandb_enable == True:
        wandb.log({"output_image[%d]"%index: wandb.Image(pic)})

def SMGAN_generate_word(Gs, opt, num_samples=10, wandb_enable=True):

    lib = Lang("song")

    if opt.input_dir == 'array':
        real_ = functions.load_phrase_from_npy(opt)
    if opt.input_dir == 'midi':
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt)
    if opt.input_dir == 'pianoroll':
        real_ = functions.load_phrase_from_npz(opt)#原 5
    if opt.input_dir == 'JSB-Chorales-dataset':
        real_ = functions.load_phrase_from_pickle(opt)

    real_ = midi2np(opt)
    real_ = midiArrayReshape(real_, opt)

    print("Input real_ shape = ", real_.shape)

    lib.addSong(real_) # for generating lib
    print("Total words = ", lib.n_words)
    opt.nword = lib.n_words
    reals = []
    reals = functions.get_reals(real_, reals, 16, [4,8,16])
    reals_num = list(map(lambda x:lib.song2num(x), reals))
    reals_num = list(map(lambda x: functions.np2torch(x), reals_num)) # track, bar, time

    if opt.mode == 'train':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_phrase[:-4], 0)
    else:
        dir2save = functions.generate_dir2save(opt)

    try:
        os.makedirs(dir2save)
        os.makedirs(dir2save+"/16th/")
        os.makedirs(dir2save+"/8th/")
        os.makedirs(dir2save+"/4th/")
    except OSError as e:
        print(e)
        exit(0)
        pass

    for ii in tqdm(range(0, num_samples, 1)):
        nbar = 32
        # din = torch.randint(1, (1, opt.ntrack, 4), dtype=torch.long).to("cuda")
        random_start = torch.randint(0, 8, [1]).item()
        din = reals_num[0][:, random_start:random_start+1, 0:4]
        din = din.reshape([1, opt.ntrack, -1])
        in_4th = din
        G_z = din

        song4th = torch.zeros([1, opt.ntrack, nbar*4], dtype=torch.long)
        song8th = torch.zeros([1, opt.ntrack, nbar*8], dtype=torch.long)
        song16th = torch.zeros([1, opt.ntrack, nbar*16], dtype=torch.long)
        # print("din length: ", din.shape[2])

        concat_mems = [tuple() for _ in range(len(Gs))]
        for l in range(nbar):
            for i, G in enumerate(Gs):
                G_z, new_mem = G(G_z, mode="top1", p=0.4, mems=concat_mems[i])
                concat_mems[i] = new_mem
                if i == 0:
                    in_4th = G_z
                if i == 1:
                    in_8th = G_z
                if i == 2:
                    in_16th = G_z
                if i != (len(Gs)-1):
                    cur_scale = 2
                    G_z = word_upsample(G_z, cur_scale)
            song16th[:, :, l*4*4:(l+1)*4*4] = in_16th[:, :, :]
            song8th[:, :,  l*2*4:(l+1)*2*4] = in_8th[:, :, :]
            song4th[:, :,  l*1*4:(l+1)*1*4] = in_4th[:, :, :]
            G_z = in_4th
        
        song16th = song16th.reshape([1, opt.ntrack, nbar, -1]) # [1, track, nbar, time]
        song16th = lib.num2song(song16th[0])[None, ] # [1, track, nbar, length, pitch]

        song8th = song8th.reshape([1, opt.ntrack, nbar, -1]) # [1, track, nbar, time]
        song8th = lib.num2song(song8th[0])[None, ] # [1, track, nbar, length, pitch]

        song4th = song4th.reshape([1, opt.ntrack, nbar, -1]) # [1, track, nbar, time]
        song4th = lib.num2song(song4th[0])[None, ] # [1, track, nbar, length, pitch]



        """
            | beat resolution | fs | what    | time |                                  
            |-----------------+----+---------+------|
            |               1 |  2 | 4 4th   | 2s   |
            |               2 |  4 | 8 8th   | 2s   |
            |               4 |  8 | 16 16th | 2s   |
            |               8 | 16 | 32 32th | 2s   |
        """

        save_pic_midi(song16th, dir2save+"/16th",     ii, opt, 4, False | wandb_enable)
        save_pic_midi(song8th,  dir2save+"/8th",      ii, opt, 2, False)
        save_pic_midi(song4th,  dir2save+"/4th",      ii, opt, 1, False)

