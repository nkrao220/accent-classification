import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
def clean_df(file):
    df = pd.read_csv(file, sep='\t')
    df_us = df[df['accent']=='us'].sample(16563)
    df_ind = df[df['accent']=='indian']
    df = df_us.append(df_ind)
    df.drop(['client_id', 'sentence', 'up_votes', 'down_votes', 'age', 'gender'],
        axis=1, inplace=True)
    return df

class Mfcc():

    def __init__(self, df, col):
        self.df = df
        self.col = col

    def mp3towav(self):
        for filename in tqdm(self.df[self.col]):
            pydub.AudioSegment.from_mp3("../data/clips/{}.mp3".format(filename)).export("../data/clips/wav/{}.wav".format(filename), format="wav")

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=13)
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        for wav in tqdm(self.df[self.col]):
            file_name = '../data/clips/wav/{}.wav'.format(wav)
            mfcc = self.wavtomfcc(file_name)
            list_of_mfccs.append(mfcc)
            self.list_of_mfccs = list_of_mfccs

    def resize_mfcc(self):
        self.target_size = 64
        resized_mfcc = [librosa.util.fix_length(mfcc, self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        self.X = resized_mfcc

    def label_samples(self):
        y_labels = np.array(self.df['accent'])
        y = np.where(y_labels=='us', 0, 1)
        self.y = y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.25)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.3)
        self.X_train = np.array(X_train).reshape(-1, 16, self.target_size)
        self.X_test = np.array(X_test).reshape(-1, 16, self.target_size)
        self.X_val = np.array(X_val).reshape(-1, 16, self.target_size)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)
        self.y_val = np.array(y_val).reshape(-1,1)

    def standardize_mfcc(self):
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        self.X_train_std = (self.X_train-train_mean)/train_std
        self.X_test_std = (self.X_test-train_mean)/train_std
        self.X_val_std = (self.X_val-train_mean)/train_std

    def oversample(self):
        temp = pd.DataFrame({'mfcc_id':range(self.X_train_std.shape[0]), 'accent':self.y_train.reshape(-1)})
        temp_1 = temp[temp['accent']==1]
        idx = list(temp_1['mfcc_id'])*3
        idx = idx + list(temp_1.sample(frac=.8)['mfcc_id'])
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        self.y_train = np.vstack((self.y_train, np.ones(232).reshape(-1,1)))

    def save_mfccs(self):
        np.save('X_train_moz.npy', self.X_train_std)
        np.save('X_test_moz.npy', self.X_test_std)
        np.save('X_val_moz.npy', self.X_val_std)
        np.save('y_train_moz.npy', self.y_train)
        np.save('y_test_moz.npy', self.y_test)
        np.save('y_val_moz.npy', self.y_val)

# 354, 293, 61
if __name__ == '__main__':
    df = clean_df('../data/validated.tsv')
    mfcc = Mfcc(df, 'path')
    mfcc.mp3towav()
    mfcc.create_mfcc()
    mfcc.resize_mfcc()
    mfcc.label_samples()
    mfcc.split_data()
    mfcc.standardize_mfcc()
    # mfcc.oversample()
    mfcc.save_mfccs()
