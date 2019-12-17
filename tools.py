import os
import librosa
import numpy as np
import pandas as pd

def load_AESDD(sr = 22050):
    """ 
        Load A.E.S.D.D dataset.
        [Input]: Sampling rate. Wav files are resampled to the specified
                 sampling rate.

        [Output]: A list of (audio_samples_array, emotion_class) tuples.
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

def create_features_label_dataframe():
    """
        Computes the features/label for each utterance.

        [Input]: None.

        [Output]: A features/label dataframe.

    """

    audio = load_AESDD()
    sentiments = pd.Series([k[1] for k in audio])
    feature_matrix = []
    for k in audio:
        sample = k[0]
        feature_matrix.append(features(sample))
    
    feature_matrix = np.array(feature_matrix)
    data_labels = (
        ['zcr_mean', 'zcr_std'] +
        ['rms_mean', 'rms_std'] +
        ['rf30_mean', 'rf30_std'] +
        ['rf50_mean', 'rf50_std'] +
        ['rf70_mean', 'rf70_std'] +
        ['rf85_mean', 'rf85_std'] +
        ['mfcc%s_mean' %(k + 1) for k in range(13)]+
        ['mfcc%s_std' %(k + 1) for k in range(13)]
        )
    data = pd.DataFrame(feature_matrix, columns = data_labels)
    data['sentiment'] = sentiments
    return data

def features(sample, sr = 22050, frame_length = 2048, hop_length = 1024):
    """
        Compute ZCR, RMS, MFCC, Rolloff Frequency Features
        for a single utterance.
    
        [Input]: 
             sample -> A single utterance (1d array)
             sr -> sampling rate
             num_frames -> number of desired frames.

        [Output]: The feature vector for this audio sample. 


        [Notes]: Frame overlapping is set to 0.5 by default.

    """
    
    # Compute Features For All Frames
    zcr = librosa.feature.zero_crossing_rate(sample, frame_length, 
                                             hop_length)


    rms = librosa.feature.rms(sample, frame_length = frame_length, 
                              hop_length = hop_length)

    rolloff_30 = librosa.feature.spectral_rolloff(sample, n_fft = frame_length, 
                                                  hop_length = hop_length, roll_percent = 0.3)

    rolloff_50 = librosa.feature.spectral_rolloff(sample, n_fft = frame_length, 
                                                  hop_length = hop_length, roll_percent = 0.5)

    rolloff_70 = librosa.feature.spectral_rolloff(sample, n_fft = frame_length, 
                                                  hop_length = hop_length, roll_percent = 0.7)

    rolloff_85 = librosa.feature.spectral_rolloff(sample, n_fft = frame_length, 
                                                  hop_length = hop_length, roll_percent = 0.85)


    mfcc = librosa.feature.mfcc(sample, n_fft = frame_length, 
                                hop_length = hop_length, n_mfcc = 13)


    feature_vector = np.array([np.mean(zcr), np.std(zcr),
                               np.mean(rms), np.std(rms),
                               np.mean(rolloff_30), np.std(rolloff_30),
                               np.mean(rolloff_50), np.std(rolloff_50),
                               np.mean(rolloff_70), np.mean(rolloff_70),
                               np.mean(rolloff_85), np.mean(rolloff_85),
                               *np.mean(mfcc, axis = 1), *np.std(mfcc, axis = 1)])
    return feature_vector



######################### Test Zone #########################
if __name__ == '__main__':
   pass 
