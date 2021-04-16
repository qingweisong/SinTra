import argparse
import torch
from SMGAN.utils import Lang
import numpy as np
import pretty_midi
import math
import pypianoroll as ppr
import collections
import torch.nn.functional as F

def midi2np_reshape(path, fs=8, all=False):
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
    if all == True:
        return data[:, :, :, :] #[track, all_bar, time, pitch]
    else:
        return data[:, 0:12, :, :] #[track, all_bar, time, pitch]


def get_kldiv_2(midiA, midiB, libB):
    """
        [track, nbarm, 16]
        [track, nbarm, 16]
        libB
    """

    ntrack, nbar, length = midiB.shape

    # cover
    all_cover = []
    for i in range(ntrack):
        num_pitch_A = len(set(midiA[i, :, :].flatten().numpy()))
        num_pitch_B = len(set(midiB[i, :, :].flatten().numpy()))
        cover = abs(num_pitch_A - num_pitch_B)
        all_cover.append(cover)
    cover_score = sum(all_cover) / len(all_cover)

    # kl
    all_kl = []
    for i in range(ntrack):
        distribute_A = [1e-5] * libB.n_words
        distribute_B = [1e-5] * libB.n_words
        for v in set(midiA[i, :, :].flatten().numpy()):
            distribute_A[int(v)] += 1
        for v in set(midiB[i, :, :].flatten().numpy()):
            distribute_B[int(v)] += 1
        kl = F.kl_div(F.log_softmax(torch.tensor(distribute_A)[None,], dim=-1), F.softmax(torch.tensor(distribute_B)[None,], dim=-1), reduction="sum")
        all_kl.append(kl.item())
    kl_score = sum(all_kl) / len(all_kl)

    return kl_score, cover_score


def get_kldiv(pathA, pathB):
    """
        A is generate midi
        B is origin midi
    """

    libA = Lang("A")
    libB = Lang("B")

    a = midi2np_reshape(pathA, all=False)
    libA.addSong(a)
    generate_word_num = libA.n_words

    b = midi2np_reshape(pathB, all=False)
    libB.addSong(b)
    origin_word_num = libB.n_words

    cover = generate_word_num*1.0 / origin_word_num

    pA = {}
    sumA = [v for i, v in libA.word2count.items()]
    sumA = sum(sumA)
    for i, v in libA.word2count.items():
        pA[i] = v*1.0

    pB = {}
    sumB = [v for i, v in libB.word2count.items()]
    sumB = sum(sumB)
    for i, v in libB.word2count.items():
        pB[i] = v*1.0

    lib_total = Lang("total")
    lib_total.index2word = libA.index2word
    lib_total.word2index = libA.word2index
    lib_total.n_words = libA.n_words
    for k in libB.word2index:
        if k not in lib_total.word2index:
            lib_total.word2index[k] = lib_total.n_words
            lib_total.index2word[lib_total.n_words] = k
            lib_total.n_words += 1

    numA = lib_total.song2num(a)
    numB = lib_total.song2num(b)

    if numA.shape[0] != numB.shape[0]:
        print("track mismatch")
        print(numA.shape[0], "  <>  " ,numB.shape[0])
        return -1, cover

    kl_track = []
    for i in range(numA.shape[0]):
        
        countA = [0] * lib_total.n_words
        tmpA = numA[i].reshape(-1)
        for v in tmpA:
            countA[int(v)] += 1.0

        countB = [0] * lib_total.n_words
        tmpB = numB[i].reshape(-1)
        for v in tmpB:
            countB[int(v)] += 1.0
        kl = F.kl_div(F.log_softmax(torch.tensor(countA)[None,], dim=-1), F.softmax(torch.tensor(countB)[None,], dim=-1), reduction="sum")
        kl_track.append(kl.item())
    return sum(kl_track) / len(kl_track), cover


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--A', help='a midi path', required=True)
    parser.add_argument('--B', help='b midi path', required=True)
    opt = parser.parse_args()

    print(get_kldiv(opt.A, opt.B))