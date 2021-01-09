import muspy
import csv
import numpy as np
import argparse
import os


def find_all_file(path):
    allfile = os.listdir(path)
    midis = []
    for file in allfile:
        if os.path.isdir(file):
            midis.extend(find_all_file(path + '/' + file))
        elif file.split(".")[-1] == 'mid':
            midis.append(path + "/" + file)
    print(midis)
    return midis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric all midi in the directory.')
    parser.add_argument("--path", "-p", required=True, help="input path")
    args = parser.parse_args()
    
    # get all paths
    midis = find_all_file(args.path)

    # init csv writer
    f = open("result.csv", 'w', encoding='utf-8', newline="")
    csv_w = csv.writer(f)
    csv_w.writerow([
        "pitch_range",
        "n_pitches_used",
        "n_pitch_classes_used",
        "polyphony",
        "polyphony_rate",
        "pitch_in_scale_rate",
        "scale_consistency",
        "pitch_entropy",
        "pitch_class_entropy",
        "empty_beat_rate",
        "drum_in_pattern_rate",
        "drum_pattern_consistency",
        "groove_consistency",
        "empty_measure_rate"
    ])
    score_sum = 0

    # metric
    for p in midis:
        music = muspy.read_midi(p)
        score = np.array([
            muspy.pitch_range(music),
            muspy.n_pitches_used(music),
            muspy.n_pitch_classes_used(music),
            muspy.polyphony(music),
            muspy.polyphony_rate(music, threshold=2),
            muspy.pitch_in_scale_rate(music, root=1, mode='major'),
            muspy.scale_consistency(music),
            muspy.pitch_entropy(music),
            muspy.pitch_class_entropy(music),
            muspy.empty_beat_rate(music),
            muspy.drum_in_pattern_rate(music, meter='duple'),
            muspy.drum_pattern_consistency(music),
            muspy.groove_consistency(music, measure_resolution=4096),
            muspy.empty_measure_rate(music, measure_resolution=4096)
        ])
        score_sum += score
        csv_w.writerow(list(score))
    csv_w.writerow()
    csv_w.writerow()
    csv_w.writerow(list(score_sum / len(paths)))
    f.close()
    pass