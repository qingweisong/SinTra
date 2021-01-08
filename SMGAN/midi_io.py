"""Utilities for writing piano-rolls to MIDI files.
"""
import numpy as np
from pypianoroll import Multitrack, Track

def write_midi(filepath, pianorolls, program_nums=None, is_drums=None,
               track_names=None, velocity=100, tempo=120.0, beat_resolution=24):
    """
    Write the given piano-roll(s) to a single MIDI file.

    Arguments
    ---------
    1.filepath : str
        Path to save the MIDI file.
    2.pianorolls : np.array, ndim=3
        The piano-roll array to be written to the MIDI file. Shape is
        (num_timestep, num_pitch, num_track).
    3.program_nums : int or list of int
        MIDI program number(s) to be assigned to the MIDI track(s). Available
        values are 0 to 127. Must have the same length as `pianorolls`.
    4.is_drums : list of bool
        True for drums. False for other instruments. Must have the same length as
        `pianorolls`.
    5.track_names : str
    """
    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]

    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same"
                         "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same"
                         "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(resolution=beat_resolution, tempo=tempo*1.0)
    for idx in range(pianorolls.shape[2]):
        if track_names is None:
            tmp = pianorolls[..., idx]
            below_pad =  (128 - tmp.shape[1]) // 2
            fore_pad = 128 - below_pad - tmp.shape[1]
            tmp = np.pad(tmp, ((0, 0), (below_pad, fore_pad)), "constant")
            track = Track(tmp, int(program_nums[idx]),
                          is_drums[idx])
                      
        else:
            track = Track(pianorolls[:, :, idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    multitrack.write(filepath)
    return multitrack

def save_midi(filepath, phrases, opt):
    """
    Save a batch of phrases to a single MIDI file.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : list of np.array
        Phrase arrays to be saved. All arrays must have the same shape.
    pause : int
        Length of pauses (in timestep) to be inserted between phrases.
        Default to 0.
    """

    # (phrase, tracks, 4, steps, pitch)
    # to
    # (phrase, 4, steps, pitch, tracks)画图
    phrases = phrases.transpose(0, 2, 3 ,4, 1)

    phrases = (phrases>0)

    if not np.issubdtype(phrases.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")

    reshaped = phrases.reshape(-1, phrases.shape[1] * phrases.shape[2], phrases.shape[3], phrases.shape[4])
    pad_width = ((0, 0), (0, opt.pause_between_samples), (0, 0), (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant')
    pianorolls = padded.reshape(-1, padded.shape[2], padded.shape[3])#(42*4+96, 56, 6)
    return write_midi(filepath, pianorolls, opt.program_num, opt.is_drum, tempo=opt.tempo)