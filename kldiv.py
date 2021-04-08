import argparse
import torch
from SMGAN.utils import Lang
import numpy as np
import pretty_midi
import math
import pypianoroll as ppr
import collections
import torch.nn.functional as F

def midi2np_reshape(path, fs=8):
    """
        inputs:
            filepath:   midi file path
            fs:         sample frequency

        returns:
            numpy array with (tracks, 128, time)
    """
    # Load MIDI file into PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(path) #type(pm)=pretty_mid.PrettyMIDI
    trakcs_len = len(pm.instruments)
    pad_time = math.ceil(pm.get_end_time() / 8) * 8
    print("paded time is [{}]".format(pad_time))
    total_notes = fs * pad_time
    tracks = []
    is_drum = []
    for i in range(trakcs_len):
        if pm.instruments[i].is_drum:
            print("Track [{}] is drum".format(i))
            is_drum.append(True)
            pm.instruments[i].is_drum = False#True返回全0 pianoroll
        else:
            is_drum.append(False)
        track = pm.instruments[i]
        track.notes.append(pretty_midi.Note(0, 0, pm.get_end_time(), pad_time))
        assert track.get_piano_roll(fs).shape[1] == total_notes, "note length is error"
        tracks.append(track.get_piano_roll(fs))
        tmp = track.get_piano_roll(fs)
        tmp[tmp == 0] = 127

    array = np.array(tracks)

    assert len(array.shape) == 3, "input dim isn't equal to 3 (tracks, pitch, time)"
    shape = array.shape
    #一小节2s  fs每秒采样次数
    data = array.reshape((shape[0], shape[1], -1, 4, int(4*fs*0.5)))#(6, 128, 31, 4, 96) [track, pitch, phrase, 4, time]
    data = data.transpose(2, 0, 3, 4, 1) # [phrase, track, 4, time, pitch]

    data = data.transpose(1, 0, 2, 3, 4) # [track, phrase, 4, time, pitch]
    shape = data.shape
    data = data.reshape([shape[0], shape[1]*shape[2], shape[3], shape[4]])
    #####最大尺度输入的是bool类型矩阵(0101)
    data = (data>0)
    return data[:, 0:12, :, :] #[track, all_bar, time, pitch]



def get_kldiv(pathA, pathB):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--A', help='a midi path', required=True)
    parser.add_argument('--B', help='b midi path', required=True)
    opt = parser.parse_args()

    libA = Lang("A")
    libB = Lang("B")

    a = midi2np_reshape(opt.A)
    libA.addSong(a)

    b = midi2np_reshape(opt.B)
    libB.addSong(b)

    # import ipdb; ipdb.set_trace()

    pA = {}
    sumA = [v for i, v in libA.word2count.items()]
    sumA = sum(sumA)
    print(sumA)
    for i, v in libA.word2count.items():
        pA[i] = v*1.0/sumA

    pB = {}
    sumB = [v for i, v in libB.word2count.items()]
    sumB = sum(sumB)
    print(sumB)
    for i, v in libB.word2count.items():
        pB[i] = v*1.0/sumB

    total_index2word = libA.index2word
    total_word2index = libA.word2index
    cnt = libA.n_words
    for k in libB.word2index:
        if k not in total_word2index:
            total_word2index[k] = cnt
            total_index2word[cnt] = k
            cnt += 1
    
    orderPA = [0] * (cnt)

    for k, v in pA.items():
        orderPA[total_word2index[k]] = v

    orderPB = [0] * (cnt)

    for k, v in pB.items():
        orderPB[total_word2index[k]] = v

    criterion = torch.nn.KLDivLoss()
    kl = criterion(F.log_softmax(torch.tensor(orderPA)), F.log_softmax(torch.tensor(orderPB)))

    print(orderPA)
    print(orderPA)
    print(orderPA == orderPB)
    print(kl)



    pass