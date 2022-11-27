'''categorize audio files into folders music or speech'''

#typing CTRL-Enter will run this cell

import os
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import json
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

# Path: main.py
'''categorize audio files music or speech'''

def choose_file():
    '''let the user pick a file from the input folder'''
    file_list = os.listdir('input')
    print('Choose a file from the list below:')
    for i, file in enumerate(file_list):
        print(i, file)
    print()
    choice = int(input('Enter a number: '))
    return file_list[choice]

def plot_msg():
    print("** Generating a waveform plot")
    print("** Plot will appear in a new window")
    print()
    print("** Close the plot to continue")
    print()

def plot_wav(file):
    plot_msg()
    '''plot the waveform of a wav file'''
    data, sample_rate = librosa.load(file)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(data, sr=sample_rate)
    plt.show()

def plot_log_power_specgram(file):
    plot_msg()
    '''plot the log power spectrogram of a wav file'''
    data, sample_rate = librosa.load(file)
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data))**2, ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.show()

def plot_mfcc(file):
    plot_msg()
    '''plot the MFCC of a wav file'''
    data, sample_rate = librosa.load(file)
    plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.show()

def plot_chroma_stft(file):
    plot_msg()
    '''plot the chroma stft of a wav file'''
    data, sample_rate = librosa.load(file)
    plt.figure(figsize=(12, 4))
    chroma_stft = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    librosa.display.specshow(chroma_stft, sr=sample_rate, x_axis='time')
    plt.show()

def plot_chroma_cqt(file):
    plot_msg()
    '''plot the chroma cqt of a wav file'''
    data, sample_rate = librosa.load(file)
    plt.figure(figsize=(12, 4))
    chroma_cqt = librosa.feature.chroma_cqt(y=data, sr=sample_rate)
    librosa.display.specshow(chroma_cqt, sr=sample_rate, x_axis='time')
    plt.show()

def analyze_menu():
    '''menu for analyzing audio files'''
    print('1. Plot waveform')
    print('2. Plot log power spectrogram')
    print('3. Plot MFCC')
    print('4. Plot chroma stft')
    print('5. Plot chroma cqt')
    print('6. MAKE A PREDICTION')
    print('0. Exit')
    print()
    choice = int(input('Enter a number: '))
    print()
    return choice

def predict(file):
    '''make a prediction about the file'''
    print("** Making a prediction")
    print()
    print("** This may take a few seconds")
    print()
    print(50 * "=")
    print("File: {}".format(file))
    print(50 * "-")
    y, sr = librosa.load(file)
    print("Duration: {} seconds".format(round(len(y) / sr, 3)))
    print("Sampling rate: {} Hz".format(sr))
    print("Number of samples: {}".format(len(y)))
    print("Minimum amplitude: {:.3f}".format(round(min(y), 3)))
    print("Maximum amplitude: {:.3f}".format(round(max(y), 3)))
    mean_amp_times_1000 = round(np.mean(y) * 1000, 3)
    print("Mean amplitude * 1000: {:.3f}".format(sum(y) / len(y) * 1000))
    print(50 * "=")
    print()
    if mean_amp_times_1000 > -0.2 and mean_amp_times_1000 <= 0:
        print("This is a speech file")
    else:
        print("This is a music file") 
    print()

# START OF PROGRAM

while True:

    file = choose_file()
    print()
    print('You chose', file)
    print()

    choice = analyze_menu()

    if choice == 1:
        plot_wav('input/' + file)
    elif choice == 2:
        plot_log_power_specgram('input/' + file)
    elif choice == 3:
        plot_mfcc('input/' + file)
    elif choice == 4:
        plot_chroma_stft('input/' + file)
    elif choice == 5:
        plot_chroma_cqt('input/' + file)
    elif choice == 6:
        predict('input/' + file)
    elif choice == 0:
        print('Goodbye!')
        break

    print()
    again = input('Analyze another file? (Y/n): ')
    if again.lower() == 'n':
        print('Goodbye!')
        break

print()
print("Program terminated")




