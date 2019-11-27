"""
Numba was not compatible for python3.8 (for a bit) thus librosa was not possible to install.
"""

import os
import soundfile as sf
import numpy as np
from scipy import signal
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


    
def aesdd_spectrogram():
    """
        Compute the spectrogram for each AESDD utterance.

        [Input]: None
        
        [Output]: A (n_utterances, samples_per_frame, num_frames, 1) np.array 
                  representing the spectrogram of each utterance.

        [Notes]: The fifth dimentions represents the channels (used for conv2d)

    """
    # Load and subsample audio files 44kHz -> 14.7kHz
    speech_signals = []
    sentiments_dir = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness':3, 'sadness': 4}
    dataset_path = './Acted Emotional Speech Dynamic Database/'

    for sentiment in os.listdir(dataset_path):
        for audiofile in os.listdir(dataset_path + sentiment):
            audio_path = dataset_path + sentiment + '/' + audiofile
            try:
                speech_signal, sr = sf.read(audio_path)
                speech_signal, sr = signal.decimate(speech_signal, 3), sr // 3
                speech_signals.append((speech_signal, sentiments_dir[sentiment]))
            except:
                pass #In case of broken audiofiles
     
    # Zero_padd signals to match maximum duration

    max_dur = max([len(k[0]) for k in speech_signals])

    samples = np.array([np.append(k[0], np.zeros(max_dur - len(k[0]))) 
                       for k in speech_signals]) 

    labels = np.array([k[1] for k in speech_signals]) 

    # Compute Spectrogram For Each Utterance 
    spectro = np.array([signal.spectrogram(k, nperseg = np.int(0.05 * sr))[-1] 
                        for k in samples])
    spectro = spectro.reshape((spectro.shape[0], spectro.shape[1], spectro.shape[2], 1))
    return spectro, to_categorical(labels)


def split_dataset(data, labels, train_percentage= 2 / 3):
    """
        Split Dataset regardless of its representation (features or spectro)

        [Input]: Data -> Features Or Spectrogram for each utterance
                 labels -> Sentiment class (str)
                 train_percentage -> The ratio len(X_train) / len(X_test)

        [Output]: The tuple (X_train ,X_test, Y_train, Y_test)

    """

    np.random.seed(182)        
    
    random_indices = np.random.permutation(range(len(data)))
    n_training = np.int(np.ceil(train_percentage * data.shape[0]))
    train_indices = random_indices[:n_training]
    test_indices = random_indices[n_training:]

    X_train, Y_train = data[train_indices], labels[train_indices]
    X_test, Y_test = data[test_indices], labels[test_indices]
    return X_train, X_test, Y_train, Y_test

    

    

    

