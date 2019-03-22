import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from mfcc import clean_df
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# create X and y arrays with correct shape
np.random.seed(1)
df = clean_df('speakers_all.csv')

labels = df['group'].unique()
X = np.load('{}_mfccs.npy'.format(labels[0]))
X = np.vstack(X).reshape(-1, 16, 512)
y = np.zeros(X.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('{}_mfccs.npy'.format(label))
    x = np.vstack(x).reshape(-1, 16, 512)
    X = np.vstack((X, x))
    label_temp = np.ones(y.shape[0])*(i+1)
    y = np.hstack((y, label_temp))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=
                    0.1, random_state=1, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, X_test,
                    test_size = 0.1, random_state=1, shuffle=True)
# add depth
X_train = X_train.reshape(X_train.shape[0], 16, 512, 1)
X_test = X_test.reshape(X_test.shape[0], 16, 512, 1)

# one hot encode y values
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
y_val_hot = to_categorical(y_val)




# Best so far
model = Sequential()
model.add(Conv2D(32, (1, 2), input_shape=(16, 512, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (1, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1))

model.add(Conv2D(64, (1, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))




model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=20, epochs=700, verbose=1,
            validation_data=(X_test, y_val_hot))

print(model.summary())