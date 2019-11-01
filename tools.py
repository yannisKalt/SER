import os
import librosa
import numpy as np
from scipy.signal.windows import hamming

def load_AESDD(sr = 22050):
    """ 
    Load A.E.S.D.D dataset.
        [Input]: Sampling rate. Wav files are resampled to the specified
                 sampling rate.

        Output: A list of (audio_samples_array, emotion_class) tuples.
    """
    speech_signals = [] 
    dataset_path = './Acted Emotional Speech Dynamic Database/'
    for sentiment in os.listdir(dataset_path):
        for audiofile in os.listdir(dataset_path + sentiment):
            audio_path = dataset_path + sentiment + '/' + audiofile
            try:
                speech_signal, sr = librosa.load(audio_path, sr = sr)
                speech_signals.append((speech_signal, sentiment))
            except:
                pass #In case of broken audiofiles

    return speech_signals

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
