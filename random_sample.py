from config import get_arguments
from SMGAN.manipulate import *
from SMGAN.training import *
import SMGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input phrase dir', required=True)
    parser.add_argument('--input_phrase', help='input phrase name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    dir2save = functions.generate_dir2save(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' % (opt.input_phrase, opt.gen_start_scale))
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' % (opt.input_phrase, opt.scale_h, opt.scale_v))
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        if opt.mode == 'random_samples':
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

            functions.adjust_scales2phrase(real_, opt)#返回real (max)并得到opt.scale_factor和opt.scale1
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,1,1,opt)#gen_start_scale!=0的话in_s是真图
            SMGAN_generate(Gs, Zs, reals, NoiseAmp, opt,  gen_start_scale=opt.gen_start_scale)

        elif opt.mode == 'random_samples_arbitrary_sizes':
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

            functions.adjust_scales2phrase(real_, opt)#返回real (max)  (1, 4, , , 8)并得到opt.scale_factor和opt.scale1
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            SMGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)
