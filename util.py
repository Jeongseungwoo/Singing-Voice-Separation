import librosa
from librosa.util import find_files
from librosa import load

import os, re 
import numpy as np
from config import *

def LoadAudio(file_path) :
    y, sr = load(file_path,sr=SR)
    stft = librosa.stft(y,n_fft=window_size,hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    return mag.astype(np.float32), phase

# Save Audiofile 
def SaveAudio(file_path, mag, phase) :
    y = librosa.istft(mag*phase,win_length=window_size,hop_length=hop_length)
    librosa.output.write_wav(file_path,y,SR,norm=True)
    print("Save complete!!")
    
def SaveSpectrogram(y_mix, y_inst,y_vocal, filename, orig_sr=44100) :
    y_mix = librosa.core.resample(y_mix,orig_sr,SR)
    y_vocal = librosa.core.resample(y_vocal,orig_sr,SR)
    y_inst = librosa.core.resample(y_inst,orig_sr,SR)

    S_mix = np.abs(librosa.stft(y_mix,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_inst = np.abs(librosa.stft(y_inst,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_vocal = np.abs(librosa.stft(y_vocal,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    
    norm = S_mix.max()
    S_mix /= norm
    S_inst /= norm
    S_vocal /= norm
    
    np.savez(os.path.join('./Spectrogram',filename+'.npz'),mix=S_mix,inst=S_inst ,vocal=S_vocal)
    
def LoadSpectrogram(target="vocal") :
    filelist = find_files('./Spectrogram', ext="npz")
    x_list = []
    y_list = []
    for file in filelist :
        data = np.load(file)
        x_list.append(data['mix'])
        if target == "vocal" :
            y_list.append(data['vocal'])
        else :
            y_list.append(data['inst'])
    return x_list, y_list


def Magnitude_phase(spectrogram) :
    Magnitude_list = []
    Phase_list = []
    for X in spectrogram :
        mag, phase = librosa.magphase(X)
        Magnitude_list.append(mag)
        Phase_list.append(phase)
    return Magnitude_list, Phase_list


def sampling(X_mag,Y_mag) :
    X = []
    y = []
    for mix, target in zip(X_mag,Y_mag) :
        starts = np.random.randint(0, mix.shape[1] - patch_size, (mix.shape[1] - patch_size) // SAMPLING_STRIDE)
        for start in starts:
            end = start + patch_size
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)