import muspy

music = muspy.read_midi("./training_data/midi/I_Wont_Let_You_Down.mid")

score = []


score.append([
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

print(score)