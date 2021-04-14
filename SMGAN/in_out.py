import pretty_midi
import numpy as np
import math
import pypianoroll as ppr
from .functions import *
import matplotlib.pyplot as plt


def array2png(roll, filepath):
    roll = roll.transpose(0, 1, 4, 2, 3)
    roll = roll.reshape(roll.shape[1], -1, roll.shape[2])#时间并称1维(t, time, pitch)
    multiTracks = ppr.Multitrack()
    for i in range(roll.shape[0]):
        pianoroll = ((roll[i, :, :]  > 0) * 100) #True*100   响度设为100
        pianoroll = torch.from_numpy(pianoroll)
        pad = nn.ZeroPad2d(padding=(0, 128-pianoroll.shape[1], 0, 0))
        pianoroll = pad(pianoroll)
        pianoroll = pianoroll.numpy()
        # print("######### show piano roll picture")
        # print(pianoroll.shape)
        track = ppr.Track(pianoroll=pianoroll)
        multiTracks.tracks.append(track)
    fig, axs = multiTracks.plot()
    plt.savefig(filepath)


def midi2np(opt):
    """
        inputs:
            filepath:   midi file path
            fs:         sample frequency

        returns:
            numpy array with (tracks, 128, time)
    """
    # Load MIDI file into PrettyMIDI object
    pm = pretty_midi.PrettyMIDI('training_data/%s/%s' % (opt.input_dir, opt.input_phrase))#type(pm)=pretty_mid.PrettyMIDI
    trakcs_len = len(pm.instruments)
    pad_time = math.ceil(pm.get_end_time() / 8) * 8
    print("paded time is [{}]".format(pad_time))
    total_notes = opt.fs * pad_time
    vel_max = []
    vel_min = []
    tracks = []
    is_drum = []
    program_num = []
    for i in range(trakcs_len):
        if pm.instruments[i].is_drum:
            print("Track [{}] is drum".format(i))
            is_drum.append(True)
            pm.instruments[i].is_drum = False#True返回全0 pianoroll
        else:
            is_drum.append(False)
        track = pm.instruments[i]
        program_num.append(track.program)
        track.notes.append(pretty_midi.Note(0, 0, pm.get_end_time(), pad_time))
        assert track.get_piano_roll(opt.fs).shape[1] == total_notes, "note length is error"
        tracks.append(track.get_piano_roll(opt.fs))
        tmp = track.get_piano_roll(opt.fs)
        vel_max.append(tmp.max())
        tmp[tmp == 0] = 127
        vel_min.append(tmp.min())

    opt.is_drum = is_drum
    opt.program_num = program_num
    opt.vel_max = vel_max
    opt.vel_min = vel_min
    return np.array(tracks)


def midiArrayReshape(array, opt):
    """
        inputs:
            array:      numpy array (tracks, 128, time)

        returns:
            numpy array with (tracks, 128, 4, 4*opt.fs, -1)

    """
    assert len(array.shape) == 3, "input dim isn't equal to 3 (tracks, pitch, time)"
    shape = array.shape
    #一小节2s  fs每秒采样次数
    data = array.reshape((shape[0], shape[1], -1, 4, int(4*opt.fs*0.5)))#(6, 128, 31, 4, 96) [track, pitch, phrase, 4, time]
    data = data.transpose(2, 0, 3, 4, 1) #[phrase, track, 4, time, pitch]
    data = data.transpose(1, 0, 2, 3, 4) # [track, phrase, 4, time, pitch]
    shape = data.shape
    data = data.reshape([shape[0], shape[1]*shape[2], shape[3], shape[4]])
    #####最大尺度输入的是bool类型矩阵(0101)
    data = (data>0)
    return data[:, 0:12, :, :] #[track, all_bar, time, pitch]


def piano_roll2midifile(piano_roll, filepath, opt):
    # print("##### write midi ")
    # print(piano_roll.shape)
    shape = piano_roll.shape
    piano_roll = piano_roll.reshape(shape[1], shape[4], shape[2]*shape[3])
    tracks_num = piano_roll.shape[0]
    new = pretty_midi.PrettyMIDI()
    for i in range(tracks_num):
        new.instruments.append(piano_roll_to_pretty_midi(piano_roll[i], opt, i))
    new.write(filepath)
    return new


def piano_roll_to_pretty_midi(piano_roll, opt, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    opt.fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./opt.fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time // opt.fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm.instruments[0]


if __name__ == '__main__':

    # data = midi2np_sqw("/home/sqw/sqw/sigle-musegan/training_data/midi/I_Wont_Let_You_Down.mid", 48)
    # data = midiArrayReshape(data)
    # array2png(data)
    pass
