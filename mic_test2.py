import os
import glob
import numpy as np
import librosa
import librosa.display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchsummary import summary

import time, logging
import threading
import collections
import queue
import os, os.path
import numpy as np
import pyaudio
from scipy import signal

import soundfile as sf

class AE(nn.Module):
    def __init__(self, input_channel=1):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channel, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def get_codes(self, x):
        return self.encoder(x)



class Audio(object):
    FORMAT = pyaudio.paFloat32

    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            custom_callback(in_data)
            return (None, pyaudio.paContinue)
        
        # Custom callback
        def custom_callback(in_data):
            """Push raw audio to the buffers
               One for DeepSpeech, the other for SSL
            """
            self.buffer_queue.put((in_data, self.idx))
            if self.idx > 4999:
               self.idx = 0
            self.idx += 1
        
        self.idx = 0
        self.buffer_queue = queue.Queue()
        
        self.device = 3
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        data32 = np.fromstring(string=data, dtype=np.float32)
        resample_size = int(len(data32) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data32, resample_size)
        resample32 = np.array(resample, dtype=np.float32)
        return resample32.tostring()

    def read_resampled(self):
        data, idx = self.buffer_queue.get()
        return self.resample(data=data, input_rate=self.input_rate)

    def read(self):
        data, idx = self.buffer_queue.get()
        return data, idx

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

def log_Mel_S(y):
    sr = 16000
    frame_length = 0.025
    frame_stride = 0.0126 
    # y, sr = librosa.load(wav_file, sr=sr)
    if len(y) < 16000:
        y = np.pad(y, (0,16000 - len(y)))
    elif len(y) > 16000:
        y = y[:16000]
    else:
        y = y

    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    s = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
    s = librosa.power_to_db(s, ref=np.max)
    return s



if __name__ == "__main__":
    model = AE()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load('model/ae_log_20.pt'))
    model.eval()
    mse = nn.MSELoss()

    aud = Audio(input_rate=44100)

    sounds = []
    while True:
        _data = aud.read_resampled()
        sound = np.frombuffer(_data, dtype=np.float32)
        sounds = np.concatenate((sounds, sound))
        if len(sounds) == 16000:
            # sf.write('mic_test.wav', sounds, 16000)
            # print(f"mean : {np.mean(np.abs(sounds))}")
            log_mel = log_Mel_S(sounds).reshape((1,1,40,80))
            output = model(torch.Tensor(log_mel).cuda())
            # print(f"log_mel.shape : {log_mel.shape}, output.shape : {output.shape}")
            loss = mse(torch.Tensor(log_mel).cuda(), output)
            print(f"loss : {loss.item()} \n")
            sounds = []
