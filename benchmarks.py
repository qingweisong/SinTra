# from steely_util.toolkits.data_convert import *
# from util.analysis.tonal import krumhansl_schmuckler
#import data_convert
import pretty_midi
from music21 import analysis, converter
import scipy.stats as stats
import math
import numpy as np
#from tonal import krumhansl_schmuckler


def ratio_of_empty_bars(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    empty_bars_num = 0
    for bar in range(bars_num):
        current_bar_info = bars_info_matrix[bar, :, :]
        if np.any(current_bar_info) == 0:
            empty_bars_num += 1
    return empty_bars_num / bars_num


def number_of_used_pitch_classses_per_bar(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    bars_pitch_classes_info = np.zeros(shape=(bars_num, 12))
    bars_pitches_num = []
    for bar in range(bars_num):
        bar_info = bars_info_matrix[bar, :, :]
        for time in range(16):
            for pitch in range(84):
                if bar_info[time, pitch] == 1 and bars_pitch_classes_info[bar, pitch % 12] == 0:
                    bars_pitch_classes_info[bar, pitch % 12] = 1
                else:
                    pass
    for bar in range(bars_num):
        current_bar_pitches = bars_pitch_classes_info[bar, :]
        bars_pitches_num.append(len(current_bar_pitches.nonzero()[0]))

    return np.mean(bars_pitches_num)


def calculate_tonic_distance(ori_path, new_path):
    tonic_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    tonic_dict = {}
    for i, tonic in enumerate(tonic_list):
        tonic_dict[tonic] = i

    tonic_original, mode_original = krumhansl_schmuckler(ori_path)
    tonic_original_index = tonic_dict[tonic_original]#音的序号
    if mode_original == 'minor':
        tonic_original_index = (tonic_original_index + 3) % 12

    tonic_new, mode_new = krumhansl_schmuckler(new_path)
    tonic_new_index = tonic_dict[tonic_new]
    if mode_new == 'minor':
        tonic_new_index = (tonic_new_index + 3) % 12

    distance = min(abs(tonic_original_index - tonic_new_index), 12 - abs(tonic_original_index - tonic_new_index))
    return distance


def in_scale_notes_ratio(path):
    bars_info_matrix = get_bar_info_from_midi(path)
    bars_num = bars_info_matrix.shape[0]
    all_notes_num = 0
    in_scale_notes_num = 0

    for bar in range(bars_num):
        bar_info = bars_info_matrix[bar, :, :]
        for time in range(64):
            for pitch in range(84):
                if bar_info[time, pitch] != 0:
                    all_notes_num += 1
                    if pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                        in_scale_notes_num += 1

    return in_scale_notes_num / all_notes_num

def tonal_distance_test():
    #print(calculate_tonic_distance('./TrainedModels/I_Wont_Let_You_Down_11.18/scale_factor=0.750000,alpha=100/7/4999_round.mid',
    print(calculate_tonic_distance('./Output/RandomSamples_denoise/I_Wont_Let_You_Down/gen_start_scale=0/5_5.mid',
                                   './training_data/midi/I_Wont_Let_You_Down.mid'))


def empty_bars_test():
    #path = './training_data/midi/I_Wont_Let_You_Down.mid'
    path = './Output/RandomSamples_denoise/I_Wont_Let_You_Down/gen_start_scale=0/5_5.mid'
    print(ratio_of_empty_bars(path), number_of_used_pitch_classses_per_bar(path))


def get_note_lengths(path):
    notes_length = [0 for _ in range(12)]
    pm = pretty_midi.PrettyMIDI(path)
    for instr in pm.instruments:#track
        if not instr.is_drum:
            for note in instr.notes:#遍历所有音符
                length = note.end - note.start#该音符持续时间
                pitch = note.pitch
                notes_length[pitch % 12] += length

    return notes_length#列表 每元素代表此类音持续时间

def get_weights(mode, name='ks'):
    if name == 'kk':
        a = analysis.discrete.KrumhanslKessler()
        # Strong tendancy to identify the dominant key as the tonic.
    elif name == 'ks':
        a = analysis.discrete.KrumhanslSchmuckler()
    elif name == 'ae':
        a = analysis.discrete.AardenEssen()
        # Weak tendancy to identify the subdominant key as the tonic.
    elif name == 'bb':
        a = analysis.discrete.BellmanBudge()
        # No particular tendancies for confusions with neighboring keys.
    elif name == 'tkp':
        a = analysis.discrete.TemperleyKostkaPayne()
        # Strong tendancy to identify the relative major as the tonic in minor keys. Well-balanced for major keys.
    else:
        assert name == 's'
        a = analysis.discrete.SimpleWeights()
        # Performs most consistently with large regions of music, becomes noiser with smaller regions of music.
    return a.getWeights(mode)


def krumhansl_schmuckler(path):
    note_lengths = get_note_lengths(path)
    key_profiles = [0 for _ in range(24)]

    for key_index in range(24):

        if key_index // 12 == 0:#<12
            mode = 'major'
        else:#12<= <24
            mode = 'minor'
        weights = get_weights(mode, 'kk')#选择一种调性特征权重
        #对音符累计总时长进行调整，将当前假设的主音排在列表的首位
        current_note_length = note_lengths[key_index:] + note_lengths[:key_index]
        #与调性分析权重计算得到皮尔逊积矩相关系数，总共得到24个数值
        pearson = stats.pearsonr(current_note_length, weights)[0]

        key_profiles[key_index] = math.fabs(pearson)
    #绝对值最大的皮尔逊积矩相关系数对应的调性，即为预测的调性
    return get_key_name(np.argmax(key_profiles))

def get_key_name(index):
    if index // 12 == 0:#index<12 mode为大调
        mode = 'major'
    else:
        mode = 'minor'

    tonic_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    tonic = tonic_list[index % 12]
    return tonic, mode

def get_bar_info_from_midi(path):
    pm = pretty_midi.PrettyMIDI(path)
    note_range = (24, 108)
    bar_num = math.ceil(pm.get_end_time() / 2)
    data = np.zeros(shape=(bar_num, 16, 84), dtype=np.float)

    # data = np.zeros((segment_num, 64, 84), np.bool_)
    quarter_length = 60 / 120 / 4
    for instr in pm.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                start = int(round(note.start / quarter_length))
                end = int(round(note.end / quarter_length))
                pitch = note.pitch
                if pitch < note_range[0] or pitch >= note_range[1]:
                    continue
                else:
                    pitch -= 24
                    for time_raw in range(start, end):
                        segment = int(time_raw / 16)
                        time = time_raw % 16
                        data[(segment, time, pitch)] = 1.0
    return data


if __name__ == '__main__':
    tonal_distance_test()#TD
    empty_bars_test()#EB      # number_of_used_pitch_classses_per_bar
    #qualified_note_ratio()#QN
