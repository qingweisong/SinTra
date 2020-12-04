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
from SMGAN.training import *
from config import get_arguments
from SMGAN.metrics import Metrics

def generate_config(opt):
    config = {}

    config['scale_mask'] = list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]))
    config['tonal_matrix_coefficient'] = (1., 1., .5)
    config['tonal_distance_pairs'] = [(i, j) for i in range(7) for j in range(7)]
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
        mat_path = os.path.join('.', filename+'.npy')
        a, b = self.metric.eval(reshaped, mat_path=None)
        self.score_matrix_mean.append(a)
        self.score_pair_matrix_mean.append(b)
        return a, b
    
    def write_npy(self, path):
        # calculate average
        average_score_matrix_mean = 0
        for i in self.score_matrix_mean:
            average_score_matrix_mean += i
        average_score_matrix_mean /= (len(self.score_matrix_mean)*1.0)

        average_score_pair_matrix_mean = 0
        for i in self.score_pair_matrix_mean:
            average_score_pair_matrix_mean += i
        average_score_pair_matrix_mean /= (len(self.score_matrix_mean)*1.0)


        if not path.endswith(".npy"):
            path = path + '.npy'
        info_dict = {
            'score_matrix_mean': self.score_matrix_mean,
            'score_pair_matrix_mean': self.score_pair_matrix_mean,
            'average_score_matrix_mean': average_score_matrix_mean,
            'average_score_pair_matrix_mean': average_score_pair_matrix_mean
            }
        print('[*] Saving score matrices...')
        np.save(path, info_dict)
        print("Successfully saved to", path)

def SMGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=5):
    config = generate_config(opt)
    metric = Evaluation(config)

    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
        in_s = dim_transformation_to_4(in_s)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
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
                I_prev = imresize(dim_transformation_to_5(I_prev, opt),1/opt.scale_factor, opt)
                #I_prev = imresize(dim_transformation_to_5(I_prev, opt),1/opt.scale_factor, opt, is_net = True)
                I_prev = dim_transformation_to_4(I_prev)[:, :, 0:round(scale_v * 4 * reals[n].shape[3]), 0:round(scale_h * reals[n].shape[4])]
                I_prev = m(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)#tensor 4

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
                denoise = functions.denoise_5D(test_round)
                save_image('%s/%d.png' % (dir2save, i), test_round, (1,1))
                save_midi('%s/%d.mid' % (dir2save, i), test_round, opt)

                save_image('%s/%d_denoise.png' % (dir2save, i), denoise, (1,1))
                save_midi('%s/%d_denoise.mid' % (dir2save, i), denoise, opt)


                #config = generate_config(opt)
                #metric = Evaluation(config)
                a, b = metric.run_eval(test_round, str(i))

                print("******************** in track ********************")
                metric.metric.print_metrics_mat(a)
                
                print("******************** track vs. track ********************")
                metric.metric.print_metrics_pair(b)


                #metric.run_eval(test_round)
            images_cur.append(I_curr)
        n+=1
    metric.write_npy("total.npy")
    return I_curr.detach()
