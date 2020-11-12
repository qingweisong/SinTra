from config import get_arguments
from SMGAN.manipulate import *
from SMGAN.training import *
import SMGAN.functions as functions
import numpy as np
from SMGAN.image_io import *


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input phrase dir', required=True)
    parser.add_argument('--input_phrase', help='input phrase name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()

    # init fixed parameters
    opt = functions.post_config(opt)
    #生成路径
    dir2save = functions.generate_dir2save(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []#?

    # if (os.path.exists(dir2save)):#若路径存在
    #     print('trained model already exist')
    # else:
    #     try:
    #         os.makedirs(dir2save)
    #     except OSError:
    #         pass

    print('********************Loading data!********************')
    #(phrase, track, 4, time, pitch)
    if opt.input_dir == 'array':
        real_ = functions.load_phrase_from_npy(opt)
    if opt.input_dir == 'midi':
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt)
    if opt.input_dir == 'pianoroll':
        real_ = functions.load_phrase_from_npz(opt)

    opt.ntrack = real_.shape[1]
    opt.npitch = real_.shape[4]
    opt.tempo = 120

    print("The num of instruments = %d" % opt.ntrack)
    print("The num of pitch = %d" % opt.npitch)
    print("The tempo of music = %d" % opt.tempo)
    print("The max of velocity = %d" % max(opt.vel_max))
    print("The min of velocity = %d" % min(opt.vel_min))



    #print(real_.shape)
    print('Training set size: %d' % real_.shape[0])
    functions.adjust_scales2phrase(real_, opt)#返回real (max)  (1, 4, , , 8)并得到opt.scale_factor和opt.scale1
    train(opt, Gs, Zs, reals, NoiseAmp)
    SMGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
