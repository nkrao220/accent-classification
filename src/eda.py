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

def clean_df(file):
    df = pd.read_csv(file)
    df_clean = df.drop(df[df['file_missing?']==True].index)
    df_clean.drop(['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], axis=1, inplace=True)
    mask = df_clean['country'].isin(['jamaica', 'usa', 'uk', 'india', 'canada',
    'philippines', 'singapore', 'malaysia', 'new zealand', 'south africa',
    'zimbabwe', 'namibia', 'pakistan', 'sri lanka', 'australia'])
    short_df = df_clean[mask]
    short_df.drop('file_missing?', axis=1, inplace= True)
    short_df['group'] = short_df.loc[:, 'country']
    short_df['group'].replace('pakistan', 'india', inplace=True)
    short_df['group'].replace('sri lanka', 'india', inplace=True)
    short_df['group'].replace('zimbabwe', 'south africa', inplace=True)
    short_df['group'].replace('namibia', 'south africa', inplace=True)
    short_df['group'].replace('jamaica', 'bermuda', inplace=True)
    short_df['group'].replace('trinidad', 'bermuda', inplace=True)
    short_df.drop(1771, inplace=True)
    group_mask = short_df['group'].isin(['usa', 'india', 'uk'])
    short_df = short_df[group_mask]
    return short_df

class Mfcc():

    def __init__(self, df, col):
        self.df = df
        self.col = col

    def mp3towav(self):
        uk = self.df[self.df['group']=='uk']
        for filename in tqdm(uk[self.col]):
            pydub.AudioSegment.from_mp3("recordings/{}.mp3".format(filename)).export("recordings/wav/{}.wav".format(filename), format="wav")


    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=13)
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        uk = self.df[self.df['group']=='uk']
        for wav in tqdm(uk[self.col]):
            file_name = 'recordings/wav/{}.wav'.format(wav)
            mfcc = self.wavtomfcc(file_name)
            list_of_mfccs.append(mfcc)
        self.list_of_mfccs = list_of_mfccs

    def resize_mfcc(self):
        self.target_size = 512
        resized_mfcc = [librosa.util.fix_length(mfcc, self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        self.X = resized_mfcc
        np.save('uk_mfccs.npy', self.X)

    def label_samples(self):
        y_labels = np.array(self.df['group'])
        y = []
        for label in y_labels:
            if label == 'usa':
                y.append(0)
            elif label == 'india':
                y.append(1)
            else:
                y.append(2)
        # np.where(y_labels=='usa', 0, 1)
        self.y = np.array(y)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.3)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.4)
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
        temp = pd.DataFrame({'mfcc_id':range(self.X_train_std.shape[0]), 'group':self.y_train.reshape(-1)})
        temp_1 = temp[temp['group']==1]
        temp_2 = temp[temp['group']==2]
        idx = list(temp_1['mfcc_id'])*4
        # idx = idx + list(temp_1.sample(frac=.8)['mfcc_id'])
        idx = idx + list(temp_2['mfcc_id'])*5
        # idx = idx + list(temp_2.sample(frac=.8)['mfcc_id'])
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        self.y_train = np.vstack((self.y_train, np.ones(62*4).reshape(-1,1)))
        self.y_train = np.vstack((self.y_train, np.ones(50*5).reshape(-1,1)*2))


    def save_mfccs(self):
        np.save('X_train_3.npy', self.X_train_std)
        np.save('X_test_3.npy', self.X_test_std)
        np.save('X_val_3.npy', self.X_val_std)
        np.save('y_train_3.npy', self.y_train)
        np.save('y_test_3.npy', self.y_test)
        np.save('y_val_3.npy', self.y_val)

# 354, 293, 61
if __name__ == '__main__':
    df = clean_df('speakers_all.csv')
    print(df['group'].value_counts())
    mfcc = Mfcc(df, 'filename')
    # mfcc.mp3towav()
    mfcc.create_mfcc()
    mfcc.resize_mfcc()
    # mfcc.label_samples()
    # y=mfcc.y
    # print(y.shape)
    # mfcc.split_data()
    # mfcc.standardize_mfcc()
    # mfcc.oversample()
    # mfcc.save_mfccs()
