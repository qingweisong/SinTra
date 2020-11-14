import numpy as np
import math

def resize_0(image_5d, scale):
    """
    This function will cut pitch near the mean of picth
    
    Inputs:
        image_5d: 5d (1, 6, 4, 96, 128)
        scale: <1
    Outputs:
        images after scale
    """
    assert image_5d.shape[0] == 1, "input phrase amount =/= 1"
    shape = image_5d.shape
    reserve_pitch = round(shape[4] * scale)
    dedim_image = image_5d.reshape(shape[0], shape[1], -1, shape[4])
    track_picth_mean = []
    scale_pitch_image = np.zeros((shape[0], shape[1], shape[2], shape[3], round(shape[4] * scale)))
    result = np.zeros((shape[0], shape[1], shape[2], round(shape[3] * scale), round(shape[4] * scale)))

    # resize on pitch 
    for i in range(shape[1]):
        pitch_mean = dedim_image[:, i, :, :].nonzero()[2]
        if len(pitch_mean) == 0:
            pitch_mean = 64
        else:
            pitch_mean = round(pitch_mean.mean())
        scale_pitch_image[:, i, :, :, :] = image_5d[:, i, :, :, int(pitch_mean-int(0.5*reserve_pitch)):math.floor(pitch_mean+int(0.5 * reserve_pitch))]

    # resize on step
    for track in range(result.shape[1]):
        for bar in range(4):
            for y in range(result.shape[4]):
                for x in range(result.shape[3]):
                    x_ = np.clip(round(x/scale), 0, scale_pitch_image.shape[3])
                    result[:, track, bar, x, y] = scale_pitch_image[:, track, bar, x_, y]

    print(result.shape)
    return result

