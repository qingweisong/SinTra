import sys
import math
import pretty_midi
from SMGAN.image_io import *
from SMGAN.midi_io import *
from SMGAN.functions import *

import argparse

def denoise_midi(filepath, fs = 48, nbar=4):
    # ==================================
    # get midi np
    # ==================================
    pm = pretty_midi.PrettyMIDI(filepath)#type(pm)=pretty_mid.PrettyMIDI
    trakcs_len = len(pm.instruments)
    pad_time = math.ceil(pm.get_end_time() / 8) * 8
    print("paded time is [{}]".format(pad_time))
    total_notes = fs * pad_time
    vel_max = []
    vel_min = []
    tracks = []
    is_drum = []
    program_num = []
    for i in range(trakcs_len):
        if pm.instruments[i].is_drum:
            print("Track [{}] is drum".format(i))
            is_drum.append(True)
        else:
            is_drum.append(False)
        track = pm.instruments[i]
        program_num.append(track.program)
        track.notes.append(pretty_midi.Note(0, 0, pm.get_end_time(), pad_time))
        assert track.get_piano_roll(fs).shape[1] == total_notes, "note length is error"
        tracks.append(track.get_piano_roll(fs))
        tmp = track.get_piano_roll(fs)
        vel_max.append(tmp.max())
        tmp[tmp == 0] = 127
        vel_min.append(tmp.min())

    array = np.array(tracks)

    # ==================================
    # reshape
    # ==================================

    assert len(array.shape) == 3, "input dim isn't equal to 3 (tracks, pitch, time)"
    shape = array.shape
    #一小节2s  fs每秒采样次数
    data = array.reshape((shape[0], shape[1], -1, nbar, int(nbar*fs*0.5)))#(6, 128, 31, 4, 96)
    data = data.transpose(2, 0, 3, 4, 1)
    data = data[0:1, :, :, :, :]
    #####最大尺度输入的是bool类型矩阵(0101)

    # bool !!!!
    # data = (data>0)

    # ==================================
    #  denoise
    # ==================================

    denoise_data = denoise_5D(data)


    # ==================================
    # save_image
    # ==================================
    save_image(".".join(filepath.split(".")[0: -1]) + "_origin" + ".png", data, (1, -1))
    save_image(".".join(filepath.split(".")[0: -1]) + "_denoise" + ".png", denoise_data, (1, -1))


    # ==================================
    # save_midi
    # ==================================

    phrases = denoise_data
    phrases = phrases.transpose(0, 2, 3 ,4, 1)

    phrases = (phrases>0)

    if not np.issubdtype(phrases.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")

    reshaped = phrases.reshape(-1, phrases.shape[1] * phrases.shape[2], phrases.shape[3], phrases.shape[4])
    pad_width = ((0, 0), (0, 96), (0, 0), (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant')
    pianorolls = padded.reshape(-1, padded.shape[2], padded.shape[3])#(42*4+96, 56, 6)
    write_midi(".".join(filepath.split(".")[0:-1]) + "_denoise" + ".mid", pianorolls, program_num, is_drum, tempo=120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='denoise a midi')
    parser.add_argument('--filepath', type=str, help='file path of midi file', required=True)
    args = parser.parse_args(sys.argv[1:])
    denoise_midi(args.filepath)
    print("Denoise Over!")
    pass
