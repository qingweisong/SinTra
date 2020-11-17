"""Base class for models.
"""
import os.path
import numpy as np
from SMGAN.midi_io import *
from SMGAN.image_io import *
from SMGAN.in_out import *

"""Save samples to an image file (and a MIDI file)."""
def save_samples(opt, filename, samples, midi = True):
    imagepath = '%s/%s.png' % (opt.outp, filename) 
    merged = save_image(imagepath, samples, (1,1))

    if midi == True:
        midipath = '%s/%s.mid' % (opt.outp, filename) 
        save_midi(midipath, samples, opt)
    return merged

def run_sampler(opt, samples, epoch, midi = True, postfix=None):
    if postfix is None:
        filename = '%d' % epoch
    else:
        filename = '%d_%s' % (epoch, postfix)
    merged = save_samples(opt, filename, samples, midi)
    return merged


"""Run evaluation."""
def run_eval(opt, samples, feed_dict, postfix):
    result = self.sess.run(target, feed_dict)
    binarized = (result > 0)
    if postfix is None:
        filename = '%d.png' % epoch
    else:
        filename = '%d_%s.png' % (epoch, postfix)
    reshaped = binarized.reshape((-1,) + binarized.shape[2:])
    mat_path = os.path.join(self.config['eval_dir'], filename+'.npy')
    _ = self.metrics.eval(reshaped, mat_path=mat_path)

