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
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    parser.add_argument('--name', type=str, help='model name')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    dir2save = functions.generate_dir2save(opt)
    print(dir2save)
    Gs = []
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' % (opt.input_phrase, opt.gen_start_scale))
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' % (opt.input_phrase, opt.scale_h, opt.scale_v))
    else:
        # try:
        #     os.makedirs(dir2save)
        # except OSError:
        #     pass

        if opt.input_dir == 'array':
            real_ = functions.load_phrase_from_npy(opt)
        if opt.input_dir == 'midi':
            real_ = midi2np(opt)
            real_ = midiArrayReshape(real_, opt)
        if opt.input_dir == 'pianoroll':
            real_ = functions.load_phrase_from_npz(opt)
        if opt.input_dir == 'JSB-Chorales-dataset':
            real_ = functions.load_phrase_from_pickle(opt)

        opt.ntrack = real_.shape[0]
        opt.npitch = real_.shape[3]
        opt.tempo = 120

        Gs = functions.load_trained_Gs(opt)
        SMGAN_generate_word(Gs, opt, wandb_enable=False)