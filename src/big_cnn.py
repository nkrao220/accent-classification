from keras.models import load_model
import numpy as np
model = load_model('../models/final_model.h5')


length = 64
X_train = np.load('X_train_moz.npy').reshape(-1, 16, length, 1)
X_test = np.load('X_test_moz.npy').reshape(-1, 16, length, 1)
X_val = np.load('X_val_moz.npy').reshape(-1, 16, length, 1)
y_train = np.load('y_train_moz.npy')
y_test = np.load('y_test_moz.npy')
y_val = np.load('y_val_moz.npy')

callbacks = [TensorBoard(log_dir='./logs')]

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0005),
              metrics=['accuracy'])

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

plt.show();

# what really optimized my model: smaller learning rate, larger number of epochs,
#
model.save('final_binary_model.h5')
print(model.summary())
