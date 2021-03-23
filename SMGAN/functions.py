import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
from scipy.ndimage import filters, measurements, interpolation
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
import os
import random
from sklearn.cluster import KMeans
import imageio
import pypianoroll
import time
import pickle

cur_time_str = "_" + "-".join(map(str, time.localtime(time.time())[0:6]))

def load_phrase_from_npy(opt):#np.load()进来就是数组
    data = np.load('training_data/%s/%s' % (opt.input_dir, opt.input_phrase))
    #-1代表几个phrase
    data = data.reshape(-1, opt.nbar, opt.ntime, opt.npitch, opt.ntrack)#(-1, 4, 96,84, track)
    data = data[0:1, :, :, :, :]

    return data

def get_reals(song, reals, in_scale=16, out_scale=[4, 8, 16]):  # all == True
    # [track, all_bars, time, pitch]
    # return [ [1, track, ... ], ]
    track, all_bars, time, pitch = song.shape

    down_scale = in_scale / np.array(out_scale, dtype=np.int)

    for i in down_scale:
        tmp = song[:, :, ::int(i), :]
        tmp = tmp[None, :, :, :, :]
        reals.append(np2torch(tmp))
    return reals


def batchify(song, length):
    # [track, all_bars, time, pitch]
    # return [N, track, all_bars, time, pitch]
    print(song.shape)
    print(length)
    track, all_bars, time, pitch = song.shape
    src = song.reshape([-1, track, length, time, pitch])
    tgt = src[1:, :, :, :, :]
    print(src[0:1].shape)
    # return src[0:-1, :, :, :, :], tgt
    return np2torch(src[0:1, :, :, :, :]), np2torch(tgt[0:1, :, :, :, :])


def load_phrase_from_pickle(opt, all=False):
    # 1 bar has 16 step
    with open('training_data/%s/%s' % (opt.input_dir, opt.input_phrase), 'rb') as p:
        data = pickle.load(p, encoding="latin1")
        data = data['train'][0] # len = 192
        song = np.zeros([1, 4*16 * (len(data) // (4*16)), 128])
        for i in range(4*16 * (len(data) // (4*16))):
            pitch = data[i]
            song[0, i, pitch] = 1


    opt.is_drum = [False]
    opt.program_num = [1]
    opt.vel_max = [60]
    opt.vel_min = [60]

    if all == False:
        song = song.reshape([1, -1, 4, 16, 128])
        song = song.transpose(1, 0, 2, 3, 4)
        return song[1:2, :, :, :, :]
    else:
        # [track, all_bars, time, pitch]
        song = song.reshape([1, -1, 16, 128])
        return song


def load_data_from_npz(opt):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load('training_data/%s/%s' % (opt.input_dir, opt.input_phrase)) as f:
        data = np.zeros(f['shape'], np.bool_)
        data[[x for x in f['nonzero']]] = True
        data = data.reshape(-1, opt.nbar, opt.ntime, opt.npitch, opt.ntrack)#(-1, 4, 96,84, track)
        data = data[0:1, :, :, :, :]
    return data

def load_data_from_MIDI(opt):
    data = pypianoroll.load('training_data/%s/%s' % (opt.input_dir, opt.input_phrase))
    Data = []
    for i in range(opt.ntrack):
        track_data = data.tracks[i].pianoroll
        Data.append(track_data)
    #列表转数组
    Data = np.array(Data)
    Data = Data.reshape(-1, opt.nbar, opt.ntime, opt.npitch, opt.ntrack)
    Data = Data[0:1, :, :, :, :]
    return Data

def load_phrase_from_single_npz(opt):
    """Load and return the training data from a npz file (sparse format)."""
    #with np.load(filename) as f:
        # data = np.zeros(f['shape'], np.bool_)
        # data[[x for x in f['nonzero']]] = True
    data = pypianoroll.load('training_data/%s/%s' % (opt.input_dir, opt.input_phrase))

    Data = []
    for i in range(opt.ntrack):
        track_data = data.tracks[i].pianoroll
        Data.append(track_data)
    #列表转数组
    Data = np.array(Data)
    Data = Data.reshape(-1, opt.nbar, opt.ntime, opt.npitch, opt.ntrack)
    Data = Data[0:1, :, :, :, :]
    return np2torch(Data)


def convert_image_np(inp):#(1, 4, h, w, track)     (b, c, h, w)->(hwc)
    #inp = denorm(inp)
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy()

    #inp = np.clip(inp,0,1)#将数组中的元素限制在0,1之间，大于1的就使得它等于1，小于0的就使得它等于0
    return inp

def create_reals_bar_pyramid(real, reals, opt):
    _, _, bar, _, _ = real.shape
    for i in range(1, bar):
        tmp = real[:, :, 0:i, :, :]
        reals.append(np2torch(tmp))
    return reals

def creat_reals_pyramid(real,reals,opt):
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale-i)#^ (4-i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)#从小向大append
    return reals

def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outp))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outp))
    torch.save(z, '%s/z_opt.pth' % (opt.outp))

def adjust_scales2phrase(real_,opt):#real_ (1, 4, 96, 84, 8)->(1,track, 4, 96, 128)
    #原->min次数 8
    opt.num_scale = math.ceil(math.log(opt.min_size / min(real_.shape[3], real_.shape[4]), opt.scale_factor_init)) + 1
    #原-> max次数 0
    scale2stop = math.ceil(math.log(min(opt.max_size, max(real_.shape[3], real_.shape[4])) / max(real_.shape[3], real_.shape[4]), opt.scale_factor_init))
    #max-> min次数 8
    opt.stop_scale = opt.num_scale - scale2stop
    #原->max 比例scale_factor_init
    opt.scale1 = min(opt.max_size / max([real_.shape[3], real_.shape[4]]),1)
    
    #最大尺度
    real = imresize(real_, opt.scale1, opt)
    #实际缩放比例
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[3],real.shape[4])),1/(opt.stop_scale))
    return real


def generate_dir2save(opt, time=None):
    dir2save = None
    if time is not None:
        real_time = "_" + time
    else:
        real_time = opt.name
    #TrainModels/
    if (opt.mode == 'train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (
            opt.input_phrase[:-4] + real_time,
            opt.scale_factor_init,opt.alpha)
    #Output/
    elif opt.mode == 'random_samples':
        dir2save = 'Output/RandomSamples/%s/gen_start_scale=%d' % (
            opt.input_phrase[:-4] + real_time,
            opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = 'Output/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (
            opt.input_phrase[:-4] + real_time,
            opt.scale_v, 
            opt.scale_h)
    return dir2save


def load_trained_pyramid(opt, mode_='train', time=None):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    dir = generate_dir2save(opt, time)
    print(dir)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
        exit(0)
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

# init fixed parameters
def post_config(opt):
    opt.device = torch.device("cuda:0")
    opt.niter_init = opt.niter#每scale训练2000个epoch
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out = 'TrainedModels/%s/scale_factor=%f/' % (
        opt.input_phrase[:-4] + cur_time_str, 
        opt.scale_factor)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # if torch.cuda.is_available() and opt.not_cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]#5
    real_down = upsampling(dim_transformation_to_4(real), scale_v * 4 *real.shape[3], scale_h * real.shape[4])#4
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[3], real_down.shape[4])
    return in_s


def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':#round()四舍五入
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)#(1, 5, 4*96, 84)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise





#上采样
def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)#align_corners=True   corner pixel of i/o tensors are aligned
    return m(im)

#重置梯度
def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates, None)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def dim_transformation_to_4(x):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], shape[2]*shape[3], shape[4])
    return x

def dim_transformation_to_5(x, opt):
    shape = x.shape
    x = x[:, :, 0: opt.nbar * (shape[2] // opt.nbar), :]
    x = x.cpu().reshape(shape[0], shape[1], opt.nbar, shape[2]//opt.nbar, shape[3])
    return x

# ============================ imresize ============================
def norm(x):
    x = (x- 0.5) * 2
    return x.clamp(-1, 1)


def denorm(x):
    x = (x + 1)/2
    return x.clamp(0,1)


def torch2uint8(x):#np数组是unit类型
    #x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def np2torch(x):#输给pytorch的tensor是float类型 的(1, 4, h, w, track)
    #x = x/128
    x = torch.from_numpy(x)
    if (torch.cuda.is_available()):
        x = x.to(torch.device('cuda'))
    x = x.type(torch.cuda.FloatTensor)
    #x = norm(x)
    return x


#np->torch   维度不变
def imresize(im, scale, opt):
    im = imresize_in(im, scale_factor=scale)
    #im = resize_0(im, scale_factor = scale, is_net=False)
    im = np2torch(im)
    #im = im[:, :, 0:int(scale * s[2]), 0:int(scale * s[3])]
    return im


def imresize_in(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    scale_factor, output_shape = fix_size(im.shape, output_shape, scale_factor)
    #print(scale_factor)
    # (cubic, 4.0)
    method, kernel_width = {
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)#kernel = None

    #when downscaling(antialiasing=True)一般是False     scale_factor = [1,1,1,s,s]
    antialiasing *= (scale_factor[3] < 1)

    sorted_dims = np.argsort(np.array(scale_factor)).tolist()#[3,4,0,1,2]

    out_im = np.copy(im)#改变out_im的值, im的值不会变
    #print(sorted_dims)
    for dim in sorted_dims:
        if scale_factor[dim] == 1.0:#跳出, 只循环前两次次dim=3, 4
            continue

        #print(im.shape, output_shape, scale_factor)
        
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        out_im = resize_along_dim(out_im, dim, weights, field_of_view)
    #print(out_im.shape)
    return out_im


def fix_size(input_shape, output_shape, scale_factor):
    if scale_factor is not None:
        if np.isscalar(scale_factor):#判断输入参数scale1是否为一个标量
            scale_factor = [scale_factor, 1]#list没有维度概念, 数组有
        add = [1] * (len(input_shape) - len(scale_factor))#[1, 1, 1, 1]
        #scale_factor = np.array(scale_factor)
        #scale_factor = scale_factor.permute((2, 3, 0, 1, 5 ))
        scale_factor = add + scale_factor#[1, 1, 1, s, s] * input_shape
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))
    return scale_factor, output_shape
    


def resize_along_dim(im, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    return np.swapaxes(tmp_out_im, dim, 0)

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel#fixed_kernel = cubic
    kernel_width *= 1.0 / scale if antialiasing else 1.0#kernel_width =1       1/0.757

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length+1)

    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view

def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

def resize_0(image_5d, scale_factor, is_net=False):
    """
    This function will cut pitch near the mean of picth
    
    Inputs:
        image_5d: 5d (1, 6, 4, 96, 128)
        scale_factor: <1
    Outputs:
        images after scale_factor
    """
    assert image_5d.shape[0] == 1, "input phrase amount =/= 1"
    if scale_factor == 1.0:
        return image_5d
    if scale_factor < 1.0:
        shape = image_5d.shape
        reserve_pitch = math.ceil(shape[4] * scale_factor)
        dedim_image = image_5d.reshape(shape[0], shape[1], -1, shape[4])
        scale_pitch_image = np.zeros((shape[0], shape[1], shape[2], shape[3], math.ceil(shape[4] * scale_factor)))
        
        result = np.zeros((shape[0], shape[1], shape[2], math.ceil(shape[3] * scale_factor), math.ceil(shape[4] * scale_factor)))

        # resize on pitch 
        for i in range(shape[1]):
            pitch_mean = dedim_image[:, i, :, :].nonzero()[2]
            if len(pitch_mean) == 0:
                pitch_mean = 64
            else:
                pitch_mean = math.ceil(pitch_mean.mean())
            
            if math.ceil(0.5 * reserve_pitch)  > pitch_mean:
                scale_pitch_image[:, i, :, :, :] = image_5d[:, i, :, :, 0: reserve_pitch]
            elif math.ceil(0.5 * reserve_pitch) > 128 - pitch_mean:
                scale_pitch_image[:, i, :, :, :] = image_5d[:, i, :, :, 128 - reserve_pitch: ]
            else:
                scale_pitch_image[:, i, :, :, :] = image_5d[:, i, :, :, int(pitch_mean-int(0.5*reserve_pitch)):int(pitch_mean-int(0.5*reserve_pitch)) + reserve_pitch]
            

        # resize on step
        for track in range(result.shape[1]):
            for bar in range(4):
                for y in range(result.shape[4]):
                    for x in range(result.shape[3]):
                        x_ = np.clip(math.ceil(x/scale_factor), 0, scale_pitch_image.shape[3] - 1)
                        result[:, track, bar, x, y] = scale_pitch_image[:, track, bar, x_, y]
        return result
    else:
        # scale > 1.0
        shape = image_5d.shape
        reserve_pitch = math.ceil(shape[4] * scale_factor)
        dedim_image = image_5d.reshape(shape[0], shape[1], -1, shape[4])
        scale_pitch_image = np.zeros((shape[0], shape[1], shape[2], shape[3], math.ceil(shape[4] * scale_factor)))
        scale_pitch_image[:, :, :, :, math.ceil(scale_pitch_image.shape[4] * 0.5) - math.ceil(shape[4]*0.5):math.ceil(scale_pitch_image.shape[4] * 0.5) - math.ceil(shape[4]*0.5) + shape[4]] = image_5d
        if is_net:
            result = np.randn((shape[0], shape[1], shape[2], math.ceil(shape[3] * scale_factor), math.ceil(shape[4] * scale_factor)))
        else:
            result = np.zeros((shape[0], shape[1], shape[2], math.ceil(shape[3] * scale_factor), math.ceil(shape[4] * scale_factor)))
        # resize on step
        for track in range(result.shape[1]):
            for bar in range(4):
                for y in range(result.shape[4]):
                    for x in range(result.shape[3]):
                        x_ = np.clip(math.ceil(x/scale_factor), 0, scale_pitch_image.shape[3] - 1)
                        result[:, track, bar, x, y] = scale_pitch_image[:, track, bar, x_, y]
        return result


def denoise_2D(x):
    '''
    This function delete isolate point
        Inputs:
            x: 2D (time, pitch)
    '''
    shape = x.shape
    for t in range(shape[0]):#time
        for p in range(shape[1]):#pitch
            # max_len = shape[0] - 1
            # left_1 = t-1 if t-1 >0 else 0
            # right_1 = t + 1 if (t + 1) < shape[0] else max_len

            # left_2 = t-2 if t-2 >0 else 0
            # right_2 = t + 2 if (t + 2) < shape[0] else max_len

            # left_3 = t-3 if t-3 >0 else 0
            # right_3 = t + 3 if (t + 3) < shape[0] else max_len

            # if (x[t, p] > 0) & (
            #         ((x[right_1, p] > 0) | (x[left_1, p] > 0)) |
            #         ((x[right_2, p] > 0) | (x[left_2, p] > 0)) 
            #         ):
            #     pass
            # else:
            #     x[t, p] = 0

            w = 1
            val = (x[t-w: t+w+1, p] > 0).sum()
            val_left = (x[t-w: t+1, p] > 0).sum()
            val_right = (x[t: t+1+w, p] > 0).sum()
            if (x[t, p] > 0) & (val >= w):
                pass
            elif (x[t, p] == 0) & (val_left > 0) & (val_right > 0):
                x[t, p] = 60
            else:
                x[t, p] = 0

    return x


def denoise_5D(x, opt=None):
    '''
    This function for 5D denoise
        Inputs:
            x: 5D (phrase_num, tracks, nbar, 4*opt.fs *0.5, 128)
    '''
    shape = x.shape
    assert shape[0] == 1, "input phrase number =/= 1"

    if opt != None:
        drum = opt.is_drum
    else:
        drum = [False] * shape[1]

    for track in range(shape[1]):
        if drum[track] == True:
            continue
        tmp = x[0, track, :, :, :].reshape((-1, shape[4]))
        x[0, track, :, :, :] = denoise_2D(tmp).reshape((-1, shape[3], shape[4]))

    return x


# def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):#kernel = None, kernel_shift_flag = False
#     # See kernel_shift function to understand what this is
#     if kernel_shift_flag:
#         kernel = kernel_shift(kernel, scale_factor)

#     # First run a correlation (convolution with flipped kernel)
#     out_im = np.zeros_like(im)
#     for channel in range(np.ndim(im)):
#         out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

#     # Then subsample and return
#     return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
#                   np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]



# def kernel_shift(kernel, sf):
#     # There are two reasons for shifting the kernel:
#     # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
#     #    the degradation process included shifting so we always assume center of mass is center of the kernel.
#     # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
#     #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
#     #    top left corner of the first pixel. that is why different shift size needed between od and even size.
#     # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
#     # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

#     # First calculate the current center of mass for the kernel
#     current_center_of_mass = measurements.center_of_mass(kernel)

#     # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
#     wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

#     # Define the shift vector for the kernel shifting (x,y)
#     shift_vec = wanted_center_of_mass - current_center_of_mass

#     # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
#     # (biggest shift among dims + 1 for safety)
#     kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

#     # Finally shift the kernel and return
#     return interpolation.shift(kernel, shift_vec)
