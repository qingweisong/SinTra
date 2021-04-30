import pickle
import numpy as np
from SMGAN.image_io import save_image
from SMGAN.midi_io import save_midi
import argparse 
import os


# def get_song_from_pickle(path, index, song_type="train"):
#     with open(path, 'rb') as p:
#         data = pickle.load(p, encoding="latin1")
#         data = data[song_type][index] # len = 192   totoal is 229
#         song = np.zeros([1, 4*16 * (len(data) // (4*16)), 128])
#         for i in range(4*16 * (len(data) // (4*16))):
#             pitch = list(map(int, data[i]))
#             song[0, i, pitch] = 1

#     # [track, all_bars, time, pitch]
#     song = song.reshape([1, -1, 16, 128])
#     return song[:, :, :, :]

def save_song(path, song, opt):
    save_image('%s/real_scale.png' % path, song, (1, 1))
    save_midi('%s/real_scale.mid' % path, song, opt, beat_resolution=4)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.pause_between_samples = 0
    opt.program_num = [1]
    opt.is_drum = [False]
    opt.tempo = 120

    path = "./training_data/JSB-Chorales-dataset/jsb-chorales-16th.pkl"
    save_path = "./training_data/JSB_midi_all"
    with open(path, 'rb') as p:
        all_data = pickle.load(p, encoding="latin1")
        total_i = 0
        for song_type in ["train", "test", "valid"]:
            length = len(all_data[song_type])
            for index in range(length):
                print(total_i)
                data = all_data[song_type][index]
                song = np.zeros([1, 4*16 * (len(data) // (4*16)), 128])
                for i in range(4*16 * (len(data) // (4*16))):
                    pitch = list(map(int, data[i]))
                    song[0, i, pitch] = 1

                song = song.reshape([1, -1, 16, 128])[None,]
                os.makedirs(save_path + "/{}".format(total_i), exist_ok=True)
                save_song(save_path + "/{}".format(total_i), song, opt)
                total_i += 1



    # for index in range(229):
    #     print(index)
    #     song = get_song_from_pickle(path, index)[None,]
    #     os.makedirs('./training_data/JSB_midi/' + str(index), exist_ok=True)
    #     save_song("./training_data/JSB_midi/{}".format(index), song, opt)