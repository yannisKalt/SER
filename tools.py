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

