iimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from sklearn.metrics import f1_score
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
# from keras.models import load_model

np.random.seed(1)
length = 64

X_train = np.load('X_train_std.npy').reshape(-1, 16, length, 1)
X_test = np.load('X_test_std.npy').reshape(-1, 16, length, 1)
X_val = np.load('X_val_std.npy').reshape(-1, 16, length, 1)
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')
callbacks = [TensorBoard(log_dir='./logs')]

# Best so far
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(16, length, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.6))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.15))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))




model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0005),
              metrics=['accuracy'])
            # took out beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
history = model.fit(X_train, y_train, batch_size=64, epochs=130, verbose=1,
            validation_data=(X_val, y_val), callbacks=callbacks, class_weight = 'balanced')

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

training_acc = history.history['acc']
test_acc = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_acc) + 1)

# Visualize loss history

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show();

# what really optimized my model: smaller learning rate, larger number of epochs,
#
model.save('final_figs/test_2_class.h5')
print(model.summary())

# Epoch 690/700
# 131/131 [==============================] - 1s 5ms/step - loss: 0.3533 - acc: 0.8092 - val_loss: 0.9705 - val_acc: 0.7273
# score = model.evaluate(X, y)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
