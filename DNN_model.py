
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

import os


class convNet(nn.Module):
    def __init__(self, batch_size, num_spkr):
        super().__init__() # just run the init of parent class (nn.Module)

        self.batch_size = batch_size
        self.num_spkr = num_spkr
        self.E_dp_prob = 0.25
        self.aud_dp_prob = 0.4
        self.fc_dp_prob = 0.25
        
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16   
        self.num_conv_kernels_3 = 8   
        self.num_conv_kernels_4 = 1  

        self.EEG_model_init()
        self.Audio_model_init()

        x = torch.randn(self.batch_size, 192, 10).view(-1, 1, 192, 10)
        self.eeg_conv_shape = None
        self.EEG_model_convs(x)   # eeg_conv_shape gets updated 

        x = torch.randn(self.batch_size, 151, 257).view(-1, 1, 151, 257)
        self.audio_conv_shape = None
        self.Audio_model_convs(x)   # Audio_conv_shape gets updated 

        # eeg_conv_shape[0] = 2,4...32 (batch size)
        # eeg_conv_shape[1] = 32 (num_conv_kernels/ input size)
        # eeg_conv_shape[2] = 48 (seq_len)
        # eeg_conv_shape[3] = 1 (final num_electrode)

        # audio_conv_shape[0] = 2,4...32 (batch size)
        # audio_conv_shape[1] = 1 (num_conv_kernels/ input size)
        # audio_conv_shape[2] = 48 (seq_len)
        # audio_conv_shape[3] = 16 (final freq dim)

        self.lstm_hidden_size = 48
        use_bidirectional = True

        if True == use_bidirectional:
            self.num_direction = 2
            self.direction_scale = 0.5
        else:
            self.num_direction = 1
            self.direction_scale = 1 

        # 2 is given to add audio data as well
        self.lstm1 = nn.LSTM((self.eeg_conv_shape[1]+2*self.audio_conv_shape[1]*self.audio_conv_shape[3]), int(self.lstm_hidden_size*self.direction_scale), 
                             bidirectional=use_bidirectional, batch_first=True)

        tmp = self.lstm_hidden_size*self.eeg_conv_shape[2]*self.eeg_conv_shape[3]        
        self.fc1 = nn.Linear(tmp, 128) 
        self.fc1_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc2 = nn.Linear(128, 128) 
        self.fc2_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc3 = nn.Linear(128, 32) 
        self.fc3_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc4 = nn.Linear(32, self.num_spkr)        

    def EEG_model_init(self):

        self.E_conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(24,1), padding=(12,0))
        # self.E_conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(12,0), dilation=(4, 1))
        self.E_mPool1 = nn.MaxPool2d((2,1))
        self.E_conv1_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv1_dp = nn.Dropout(p=self.E_dp_prob) 
 

        self.E_conv2 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(6,0), dilation=(2, 1))
        self.E_mPool2 = nn.MaxPool2d((1,2))
        self.E_conv2_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv2_dp = nn.Dropout(p=self.E_dp_prob) 

        self.E_conv3 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,5), padding=(3,2))
        self.E_mPool3 = nn.MaxPool2d((2,5))
        self.E_conv3_bn = nn.BatchNorm2d(self.num_conv_kernels_1)   
        self.E_conv3_dp = nn.Dropout(p=self.E_dp_prob) 

        self.E_conv4 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(3,0))
        self.E_conv4_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv4_dp = nn.Dropout(p=self.E_dp_prob) 

    def Audio_model_init(self):
        self.A_conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(1,7), padding=(0,3)) 
        self.A_conv1_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.A_conv1_dp = nn.Dropout(p=self.aud_dp_prob) # ReLU before

        self.A_conv2 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(0,0)) 
        self.A_mPool2 = nn.MaxPool2d((1,4))
        self.A_conv2_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv2_dp = nn.Dropout(p=self.aud_dp_prob) # ReLU before
    
        self.A_conv3 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3,5), padding=(0,16), dilation=(8,8))
        self.A_mPool3 = nn.MaxPool2d((1,2))
        self.A_conv3_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv3_dp = nn.Dropout(p=self.aud_dp_prob) # ReLU before

        self.A_conv4 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3,3), padding=(0,16), dilation=(16,16))
        self.A_conv4_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv4_dp = nn.Dropout(p=self.aud_dp_prob) # ReLU before

        self.A_conv5 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_4, kernel_size=(1,1))
        self.A_mPool5 = nn.MaxPool2d((2,2))
        self.A_conv5_bn = nn.BatchNorm2d(self.num_conv_kernels_4)  
        self.A_conv5_dp = nn.Dropout(p=self.aud_dp_prob) # ReLU before

    def EEG_model_convs(self, x):
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.E_conv1_bn(self.E_mPool1(self.E_conv1(x)))
        x = self.E_conv1_dp(F.relu(x))

        x = self.E_conv2_bn(self.E_mPool2(self.E_conv2(x)))
        x = self.E_conv2_dp(F.relu(x))

        x = self.E_conv3_bn(self.E_mPool3(self.E_conv3(x)))
        x = self.E_conv3_dp(F.relu(x))

        x = self.E_conv4_bn(self.E_conv4(x))
        x = self.E_conv4_dp(F.relu(x))        

        if self.eeg_conv_shape is None:
            self.eeg_conv_shape = x.shape
            print(self.eeg_conv_shape)
        return x

    def Audio_model_convs(self, x):
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.A_conv1_bn(self.A_conv1(x))
        x = self.A_conv1_dp(F.relu(x))

        x = self.A_conv2_bn(self.A_mPool2(self.A_conv2(x)))
        x = self.A_conv2_dp(F.relu(x))

        x = self.A_conv3_bn(self.A_mPool3(self.A_conv3(x)))
        x = self.A_conv3_dp(F.relu(x))

        # x = self.A_conv4_bn(self.A_mPool4(self.A_conv4(x)))
        x = self.A_conv4_bn(self.A_conv4(x))
        x = self.A_conv4_dp(F.relu(x))

        x = self.A_conv5_bn(self.A_mPool5(self.A_conv5(x)))
        x = self.A_conv5_dp(F.relu(x)) 
        
        if self.audio_conv_shape is None:
            self.audio_conv_shape = x.shape
            print(self.audio_conv_shape)
        return x

    
    def forward(self, eeg_x, aud):
        
        eeg_x = self.EEG_model_convs(eeg_x)
        # for the lstm input, first dim after batch size should be seq_len 
        eeg_x = eeg_x.view(-1, self.eeg_conv_shape[2], self.eeg_conv_shape[3]*self.eeg_conv_shape[1])

        aud_x = []
        for id in range(self.num_spkr):
            tmp = aud[:, id, :, :]
            a = self.Audio_model_convs(tmp.view(-1, 1, 151, 257))

            # for the lstm input, first dim after batch size should be seq_len 
            a = a.reshape(-1, self.audio_conv_shape[2], self.audio_conv_shape[3]*self.audio_conv_shape[1])
            aud_x.append(a)
            del a
        
        
        x = torch.cat([aud_x[0], eeg_x, aud_x[1]], dim=2)

        # Input = batch, seq_len, input_size (self.eeg_conv_shape[0], self.eeg_conv_shape[2]*self.eeg_conv_shape[3], self.eeg_conv_shape[1]) 
        # Output = batch, seq_len, self.lstm_hidden_size
        # hidden state: num_layers * num_directions, batch, self.lstm_hidden_size*self.direction_scale  (read my notes from book)
        # cell state: same as hidden state
        
        x, (hidden_state, cell_state) = self.lstm1(x)
        x = x.reshape(-1, self.lstm_hidden_size*self.eeg_conv_shape[2])

        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.fc1_dp(F.relu(self.fc1(x)))
        x = self.fc2_dp(F.relu(self.fc2(x)))
        x = self.fc3_dp(F.relu(self.fc3(x)))
        x = F.softmax(self.fc4(x), dim=1) 
        
        return x
