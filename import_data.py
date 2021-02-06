
import torch 
import torchvision as tv
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from scipy import stats, signal

import numpy as np
import pandas as pd
import random
import os


class HI_Dataset(Dataset):
    def __init__(self, base_path, csv_file):
        self.base_path = base_path
        self.csv_file = csv_file

        # band pass filtering (1-9 Hz)
        self.sos_bp = ([[ 0.00432289,  0.00390431,  0.00432289,  1.        , -0.68058302,         0.18722999],
                        [ 1.        , -0.23389564,  1.        ,  1.        , -0.8336818 ,         0.39304264],
                        [ 1.        , -0.73839576,  1.        ,  1.        , -1.02270308,         0.62849011],
                        [ 1.        ,  0.        , -1.        ,  1.        , -1.20601611,         0.27919438],
                        [ 1.        , -1.99869015,  1.        ,  1.        , -1.81621689,         0.82786496],
                        [ 1.        , -0.9224206 ,  1.        ,  1.        , -1.20537201,         0.86885965],
                        [ 1.        , -1.99561804,  1.        ,  1.        , -1.86732577,         0.8802607 ],
                        [ 1.        , -1.99248629,  1.        ,  1.        , -1.91594458,         0.93054023],
                        [ 1.        , -1.99061636,  1.        ,  1.        , -1.96125717,         0.97707891]])

        # high pass filtering (1 Hz)
        self.sos_hp = ([[ 0.7648476 , -0.7648476 ,  0.        ,  1.        , -0.91094815,         0.        ],
                       [ 1.        , -1.99887104,  1.        ,  1.        , -1.83030101,         0.83930608],
                       [ 1.        , -1.99601526,  1.        ,  1.        , -1.85531953,         0.86711526],
                       [ 1.        , -1.99277274,  1.        ,  1.        , -1.89612558,         0.91128566],
                       [ 1.        , -1.99065917,  1.        ,  1.        , -1.95060444,         0.96828771]])

        self.get_data()

        
    def __getitem__(self, index):
        data = self.data[index] 
        tmp_path1 = self.base_path + '\\EEG\\' + data[0]

        X1 = np.load(os.path.normpath(os.path.join(tmp_path1 +'.npy')))
        X1 = signal.sosfiltfilt(self.sos_hp, X1, axis=0)
        X1 = stats.zscore(X1, axis=0)  

        X1 = torch.from_numpy(X1.copy()).float()        
        tmp_path2 = self.base_path + '\\Speech_data\\' + data[1]
        X2 = np.load(os.path.normpath(os.path.join(tmp_path2 +'.npy')))
        X2 = torch.from_numpy(X2).float() 
 
        label = data[2]
        label = torch.tensor(label).float()

        return (X1, X2, label)
    
    def __len__(self):
        return len(self.data)

    def get_data(self):
        tmp_filename = self.base_path + '\\' + self.csv_file 
        df = pd.read_csv(tmp_filename, header=None)
        data_list = df.values.tolist()         

        Data = []        
        for data in data_list:
            labels = np.eye(2)[data[1]] 
            labels = labels.astype(np.int64)
            eeg_file = data[0]

            d = data[0].split('_')
            audio_file = d[1]
            Data.append((eeg_file, audio_file, labels))
        self.data = Data    


# def add_awgn(ip_spec, snr_db):
#     op_spec = []

#     for ch_id in range(np.shape(ip_spec)[2]):
#         tmp_ip_spec = np.reshape(ip_spec[:, :, ch_id], np.shape(ip_spec)[0]*np.shape(ip_spec)[1])
#         sig_avg_db = 10 * np.log10(np.mean(tmp_ip_spec))
#         noise_avg_db = sig_avg_db - snr_db
#         noise_avg_power = 10 ** (noise_avg_db / 10)

#         # Generate an sample of white noise
#         mean_noise = 0
#         noise = np.random.normal(mean_noise, noise_avg_power, len(tmp_ip_spec))

#         # If original speech is used instead of spectrogram, np.sqrt(...) needs to be taken
#         # noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_power), len(tmp_ip_spec))

#         tmp_op_spec = tmp_ip_spec - noise
#         op_spec.append(np.reshape(tmp_op_spec, np.shape(ip_spec)[0]*np.shape(ip_spec)[1]))
#         del tmp_ip_spec, tmp_op_spec

#     return np.asarray(op_spec)

# In[]

# fs_final = 64
# N = 9
# Rs = 60
# fstop1 = 1
# fstop2 = 11  # First Stopband Frequency
# Wn = np.array([fstop1, fstop2])/(fs_final/2);

# sos_bp = signal.cheby2(N, Rs, Wn, 'band', analog = False, output='sos')
# w, h = signal.sosfreqz(sos_bp, worN=1500)

# plt.figure(2)
# plt.subplot(2, 1, 1)
# db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
# plt.plot(w/np.pi, db)
# plt.ylim(-75, 5)
# plt.grid(True)
# plt.yticks([0, -20, -40, -60])
# plt.ylabel('Gain [dB]')
# plt.title('Frequency Response')
# plt.subplot(2, 1, 2)
# plt.plot(w/np.pi, np.angle(h))
# plt.grid(True)
# plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
#            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
# plt.ylabel('Phase [rad]')
# plt.xlabel('Normalized frequency (1.0 = Nyquist)')
# plt.show()     

# In[]

# fs_final = 64
# N = 9
# Rs = 60
# fstop1 = 1
# Wn = np.array([fstop1])/(fs_final/2);

# sos_hp = signal.cheby2(N, Rs, Wn, 'high', analog = False, output='sos')
# w, h = signal.sosfreqz(sos_hp, worN=1500)

# plt.figure(2)
# plt.subplot(2, 1, 1)
# db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
# plt.plot(w/np.pi, db)
# plt.ylim(-75, 5)
# plt.grid(True)
# plt.yticks([0, -20, -40, -60])
# plt.ylabel('Gain [dB]')
# plt.title('Frequency Response')
# plt.subplot(2, 1, 2)
# plt.plot(w/np.pi, np.angle(h))
# plt.grid(True)
# plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
#            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
# plt.ylabel('Phase [rad]')
# plt.xlabel('Normalized frequency (1.0 = Nyquist)')
# plt.show() 