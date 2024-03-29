#!/home/sqw/anaconda3/envs/nlp/bin/python
from config import get_arguments
from SMGAN.manipulate import *
# from SMGAN.training import *
from SMGAN.training_noGAN import *
import SMGAN.functions as functions
import numpy as np
from SMGAN.image_io import *
import wandb

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input phrase dir', required=True)
    parser.add_argument('--input_phrase', help='input phrase name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--name', help='describe this time ', default='train')
    parser.add_argument('--model_type', help='describe this time ', default='transformerXL')
    parser.add_argument('--single', action="store_true", help='single mode')
    parser.add_argument('--index', type=int, help='index for train dataset', default=0)
    parser.add_argument('--topP', action="store_true", help='topP')
    opt = parser.parse_args()


    if opt.single == True:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>> Single Model >>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if opt.single == False:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>> Multi Stage Model >>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # init fixed parameters
    opt = functions.post_config(opt)
    #生成路径
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):#若路径存在
        print('trained model already exist')
        exit(0)
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

    wandb.init(
        project="dead",
        config = {
            "name": opt.name,
            "niter": opt.niter,
            "model_type": opt.model_type,
            "path": dir2save,
            "index": opt.index,
            "info": ""
        }
    )


    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []#?


    print('********************Loading data!********************')
    #(phrase, track, 4, time, pitch)
    if opt.input_dir == 'array':
        real_ = functions.load_phrase_from_npy(opt)
    if opt.input_dir == 'midi':
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt)
    if opt.input_dir == 'pianoroll':
        real_ = functions.load_phrase_from_npz(opt)
    if opt.input_dir == 'JSB-Chorales-dataset':
        real_ = functions.load_phrase_from_pickle(opt)
    else:
        real_ = midi2np(opt)
        real_ = midiArrayReshape(real_, opt)
    
    # import ipdb; ipdb.set_trace()

    opt.ntrack = real_.shape[0]
    opt.npitch = real_.shape[3]
    opt.tempo = 120

    print("The num of instruments = %d" % opt.ntrack)
    print("The num of pitch = %d" % opt.npitch)
    print("The tempo of music = %d" % opt.tempo)
    print("The program_num = {}".format(opt.program_num))
    print("The max of velocity = %d" % max(opt.vel_max))
    print("The min of velocity = %d" % min(opt.vel_min))


    # config = generate_config(opt)
    # real_metric = Evaluation(config)
    # real_metric.run_eval(real_)
    # real_metric.write_txt("real")

    #a, b = metric.run_eval(real_)
    # print("*********************** in track ***********************")
    # metric.metric.print_metrics_mat(a)
    
    # print("******************** track vs. track ********************")
    # metric.metric.print_metrics_pair(b)



    # print('Training set size: %d' % real_.shape[0])
    # functions.adjust_scales2phrase(real_, opt)#返回real (max)  (1, 4, , , 8)并得到opt.scale_factor和opt.scale1
    trainWOGAN(opt, Gs, Zs, reals, NoiseAmp, single=opt.single)
    SMGAN_generate_word(Gs, opt)

    # scale_v = 2
    # opt.gen_start_scale = 0
    # in_s = functions.generate_in2coarsest(reals, scale_v, 1,opt)
    # SMGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=scale_v, scale_h=1)
