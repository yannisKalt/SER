import os
import pandas as pd
import numpy as np
from scipy.signal.windows import hamming
from scipy.io import wavfile

def load_wavfiles():
    """ 
    Load Acted Emotional Speech Dynamic Database.
    """

    database_path = './Acted Emotional Speech Dynamic Database/'
    os.chdir(database_path)
    audio_files = []
    for sentiment in os.listdir():
        for audio_file in os.listdir(sentiment):
            try:
                audio = wavfile.read(sentiment + '/' + audio_file)
                audio_files.append([audio[1], sentiment])
            except:
                pass
    return audio_files
def frame_signal(fs, signal, frame_size = 0.025, frame_step = 0.01):
    """
    Split the signal into several frames via hamming window
    Input:
        -fs: sampling rate.
        -signal: 1-d array containing the signal samples.
        -frame_step [s]: time advancement of each frame.
        -frame_size [s]: time duration of frame.
    Output: 
        frames: Hamming Filtered signal frames.
    """
    (frame_sample_len, frame_sample_step) = (int(round(fs * frame_size)), 
                                             int(round(fs * frame_step)))

    n_frames = int(np.ceil((len(signal) - frame_sample_len) / frame_sample_step)) + 1

    ## Zero pad signal in order to fit integer number of frames. ##
    z = np.zeros(int(n_frames * frame_sample_step + frame_sample_len - len(signal)))
    signal = np.append(signal, z)
    
    ## Compute Frames With Hamming Partial Convolution ##
    start = lambda k: k * frame_sample_step
    stop = lambda k: start(k) + frame_sample_len
    frames = np.array([signal[start(k):stop(k)] for k in range(n_frames)]) 

    frames *= hamming(frame_sample_len)
    return frames
