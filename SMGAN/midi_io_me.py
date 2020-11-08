"""Utilities for writing piano-rolls to MIDI files."""
import numpy as np
from pypianoroll import Multitrack, Track


def get_midi(path, pianorolls, program_nums=None, is_drums=None, track_names=None, velocity=100, tempo=120.0, beat_resolution=24):
    """
    Write the given piano-roll(s) to a single MIDI file.

    Arguments
    ---------
    1.filepath : str
        Path to save the MIDI file.
    2.pianorolls : bool类型np.array, ndim=3
        The piano-roll array to be written to the MIDI file. Shape is
        (num_timestep, num_pitch, num_track).
    3.program_nums : int or list of int
        MIDI program number(s). Available
        values are 0 to 127. Must have the same length as `pianorolls`.
    4.is_drums : list of bool
        Drum indicator(s). True for
        drums. False for other instruments. Must have the same length as
        `pianorolls`.
    5.track_names : list of str
        Track name(s) to be assigned to the MIDI track(s).
    """
    if not np.issubdtype(pianorolls.dtype, np.bool_):#判断数组类型（dtype）是否为bool
        raise TypeError("Support only binary-valued piano-rolls")
    # if isinstance(program_nums, int):
    #     program_nums = [program_nums]
    # if isinstance(is_drums, int):
    #     is_drums = [is_drums]

    #program_nums, is_drums=None时不进入
    if len(program_nums) != pianorolls.shape[2]:#8
        raise ValueError("'pianorolls' and 'program_nums' must have the same length")
    if len(is_drums) != pianorolls.shape[2]:#8
        raise ValueError("'pianorolls' and 'is_drums' must have the same length")
    
    if program_nums is None:
        program_nums = [0] * len(pianorolls)#96
    if is_drums is None:
        is_drums = [False] * len(pianorolls)#96


    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for i in range(pianorolls.shape[2]):
        if track_names is None:
            track = Track(pianorolls[:, :, i], program_nums[i], is_drums[i])
        else:
            track = Track(pianorolls[:, :, i], program_nums[i], is_drums[i], track_names[idx])
        multitrack.append_track(track)
    multitrack.write(path)

def save_midi(opt, path, binarized_phrases):
    """
    Save a batch of phrases to a single MIDI file.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    binarized_phrases : list of np.array
        Phrase arrays to be saved. All arrays must have the same shape.
    pause : int
        Length of pauses (in timestep) to be inserted between phrases.
        Default to 0.
    """
    print(type(binarized_phrases))
    #if phrases.dtype == np.bool_:
    if not np.issubdtype(binarized_phrases.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    reshaped = binarized_phrases.reshape(1, binarized_phrases.shape[1] * binarized_phrases.shape[2], 
                                    binarized_phrases.shape[3], binarized_phrases.shape[4])#(1, 4x96, 84, 8)
    pad_width = ((0, 0), (0, opt.pause_between_samples),#phrase后的空隙
                 (opt.lowest_pitch, 128 - opt.lowest_pitch - opt.npitch),#??????????(24, 20)
                 (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant')#避免因为卷积运算导致输出图像缩小和图像边缘信息丢失，常常采用numpy.pad()进行填充操作，即在图像四周边缘填充0，使得卷积运算后图像大小不会缩小，同时也不会丢失边缘和角落的信息.
    pianorolls = padded.reshape(-1, padded.shape[2], padded.shape[3])#(4x96, 84, 8)

    get_midi(path, pianorolls, opt.programs, opt.is_drum, opt.track_names, tempo = opt.tempo, beat_resolution = opt.beat_resolution)