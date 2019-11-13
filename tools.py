import os
import librosa
import numpy as np
import pandas as pd

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

def features_to_dataframe():
    """
        Compute ZCR, RMS, MFCC, Rolloff Features
        from AESDD Dataset.

        -Input: None
        -Output: Feature-Sentiment Dataframe

    """
    audio = load_AESDD()
    sentiments = pd.Series([k[1] for k in audio])
    feature_matrix = []
    for k in audio:
        sample = k[0]
        feature_matrix.append(features(sample))
    
    feature_matrix = np.array(feature_matrix)
    data_labels = (
        ['zcr_f%s' % fn for fn in range(6)] +
        ['rms_f%s' % fn for fn in range(6)] +
        ['rf30_f%s' % fn for fn in range(6)] +
        ['rf50_f%s' % fn for fn in range(6)] +
        ['rf70_f%s' %fn for fn in range(6)] +
        ['rf85_f%s' %fn for fn in range(6)] +
        ['mfcc%s_f%s' %(k +1, fn) for k in range(13) for fn in range(6)]
        )
    data = pd.DataFrame(feature_matrix, columns = data_labels)
    data['sentiment'] = sentiments
    return data

def features(sample, sr = 22050, num_frames = 6):
    """
        Compute ZCR, RMS, MFCC, Rolloff Frequency Features
        for a single utterance.

    """
    # n: frame_length
    # h: hop_length
    # l: sample length
    # t: last frame index   

    
    t = num_frames - 1
    
    while len(sample) % (t + 2):
        sample = np.append(sample, 0)

    l = len(sample)
    n = np.int(np.ceil(2 * l / (t + 2)))
    h = n // 2
    
    # Compute Features 
    zcr = librosa.feature.zero_crossing_rate(sample, frame_length = n,
                                             hop_length = h, center = False)

    rms = librosa.feature.rms(y = sample, frame_length = n, 
                              hop_length = h, center = False)

    rolloff_30 = librosa.feature.spectral_rolloff(sample, n_fft = n, hop_length = h,
                                                  center = False, roll_percent = 0.3)

    rolloff_50 = librosa.feature.spectral_rolloff(sample, n_fft = n, hop_length = h,
                                                  center = False, roll_percent = 0.5)

    rolloff_70 = librosa.feature.spectral_rolloff(sample, n_fft = n, hop_length = h,
                                                  center = False, roll_percent = 0.7)
    
    rolloff_85 = librosa.feature.spectral_rolloff(sample, n_fft = n, hop_length = h,
                                                  center = False, roll_percent = 0.85)

    mfcc = librosa.feature.mfcc(sample, n_fft = n, hop_length = h, 
                                center = False, n_mfcc = 13)

    mfcc = np.reshape(mfcc, (1, mfcc.size)) 
    
    feature_vector = np.concatenate([zcr, rms, rolloff_30, rolloff_50, rolloff_70, 
                                    rolloff_85, mfcc], axis = None)
    return feature_vector



######################### Test Zone #########################
if __name__ == '__main__':
   pass 
