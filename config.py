import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    #load, input, save configurations:
    parser.add_argument('--model', default='', help='SMGAN/SBMGAN')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=20)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=128)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train per scale')#epoch
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)#??????????????????????????????????
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)#??????????????????????????????
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=100)

    #Data(array/pianoroll/midi)
    parser.add_argument('--nbar', type=int, default=4)
    parser.add_argument('--ntime', type=int, default=96)
    parser.add_argument('--tempo', type=int, default=-1)
    parser.add_argument('--beat_resolution', type=int, default=24)
    parser.add_argument('--lowest_pitch', type=int, default=0)
    parser.add_argument('--pause_between_samples', type=int, default=96)    
    parser.add_argument('--fs', type=int, help="sample freuency", default=48)

    #Tracks
    #parser.add_argument('--track_names', default=('Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble', 'Reed', 'Synth Lead', 'Synth Pad'))
    #parser.add_argument('--programs', default=[0, 0, 24, 32, 48, 64, 80, 88])
    #parser.add_argument('--is_drums', default=[True, False, False, False, False, False, False, False])

    # Samples
    parser.add_argument('--nsample', type=int, default=1)
    #parser.add_argument('--sample_grid', default=(1,1))


    return parser