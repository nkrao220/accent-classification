import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.preprocessing import StandardScaler


def clean_df(file):
    df = pd.read_csv(file)
    df_clean = df.drop(df[df['file_missing?']==True].index)
    df_clean.drop(['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], axis=1, inplace=True)
    mask = (df_clean['country'] == 'jamaica') | (df_clean['country'] == 'usa') | (df_clean['country'] == 'uk') | (df_clean['country'] ==
                'india') | (df_clean['country'] == 'canada') | (df_clean['country'] == 'philippines') | (df_clean['country'] ==
                'singapore') | (df_clean['country'] == 'malaysia') | (df_clean['country'] == 'new zealand') | (df_clean['country'] ==
                'south africa') | (df_clean['country'] == 'zimbabwe') | (df_clean['country'] == 'namibia') | (df_clean['country'] ==
                'pakistan') | (df_clean['country'] == 'sri lanka') | (df_clean['country'] == 'australia')
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
    short_df = short_df[(short_df['group'] == 'usa') | (short_df['group'] == 'india')]
    return short_df

def mp3tomfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr= 16000)
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=13, hop_length=1024)
    return mfcc

def create_mfcc(clean_df, col):
    mfcc_all = []
    for mp3 in clean_df[col]:
        file_name = 'recordings/wav/{}.wav'.format(mp3)
        mfcc = mp3tomfcc(file_name)
        mfcc_all.append(mfcc)
    return mfcc_all

def resize_mfcc(list_of_mfccs):
    mfcc_lens = [i.shape[1] for i in list_of_mfccs]
    target_size = 512
    adjusted_mfcc = [librosa.util.fix_length(mfcc, target_size, axis=1)
                     for mfcc in mfcc_all]
    adjusted_mfcc = [np.vstack((np.zeros((3,512)), mfcc)) for mfcc in adjusted_mfcc]
    return adjusted_mfcc

def standardize_mfcc(list_of_mfccs):
    mfcc_std = [sklearn.preprocessing.scale(mfcc, axis=1) for mfcc in list_of_mfccs]
    return mfcc_std

def balance_samples(df, undersample=True):
    labels = df['group'].unique()
    majority_class = 'usa'
    majority_size = df[df['group'] == 'us'].shape[0]
    if undersample:
        for label in labels:
            temp = pd.DataFrame({'mfcc':mfcc_resized, 'group':df['group']})
            label_list = np.array(temp['mfcc'][temp['group'] == label].sample(82))
            np.save('{}_mfccs.npy'.format(label), label_list)
    else:
        for label in labels:
            temp = pd.DataFrame({'mfcc':mfcc_resized, 'group':df['group']})
            if label == majority_class:
                label_list = np.array(temp['mfcc'][temp['group'] == label]
                np.save('{}_mfccs.npy'.format(label), label_list)
            else:
                label_list = np.array(temp['mfcc'][temp['group'] == label].sample(309))


if __name__ == '__main__':
    df = clean_df('speakers_all.csv')
    mfcc_all = create_mfcc(df, 'filename')
    print(len(mfcc_all))
    mfcc_resized = resize_mfcc(mfcc_all)
    mfcc_resized_std = standardize_mfcc(mfcc_resized)
    labels = df['group'].unique()
    for label in labels:
        temp = pd.DataFrame({'mfcc':mfcc_resized_std, 'group':df['group']})
        label_list = np.array(temp['mfcc'][temp['group'] == label].sample(82))
        np.save('{}_mfccs.npy'.format(label), label_list)
    mfcc_widths = [mfcc.shape[1] for mfcc in mfcc_resized_std]
    print(len(mfcc_resized), min(mfcc_widths), max(mfcc_widths), mfcc_resized_std[0].mean(), mfcc_resized_std[0].std())
