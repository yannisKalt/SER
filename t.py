from tools import load_AESDD
import numpy as np

sr = 16000
audio = load_AESDD(sr)
waveforms = np.array([k[0] for k in audio])
labels = np.array([k[1] for k in audio])

np.save('labels', labels)
np.save('audio', audio_waveform)

