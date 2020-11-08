"""Base class for models.
"""
import os.path
import numpy as np
from SMGAN import midi_io
from SMGAN.image_io import *
from SMGAN.in_out import *

"""Save samples to an image file (and a MIDI file)."""
def save_samples(opt, filename, samples):
    imagepath = '%s/%s.png' % (opt.outp, filename) 
    save_image(imagepath, samples, (1,1))
    #image_io.save_image(opt, imagepath, samples)
    
    binarized = (samples > 0)#再次二值化
    midipath = '%s/%s.mid' % (opt.outp, filename) 
    piano_roll2midifile(binarized, midipath, opt)
    #midi_io.save_midi(opt, midipath, binarized)


def run_sampler(opt, samples, epoch, postfix=None):
    if postfix is None:
        filename = '%d' % epoch
    else:
        filename = '%d_%s' % (epoch, postfix)
    save_samples(opt, filename, samples)

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

