import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kerascnn import cnnmodel 
import random
import cv2
from webcam import cnn_cam_mnist


(train, train_label), (test, test_label) = tf.keras.datasets.mnist.load_data()
print (test.shape, type(test_label))
reshape_train = train.reshape(train.shape[0], 28, 28, 1)
reshape_test = test.reshape(test.shape[0], 28, 28, 1)
image_index = 7 # You may select anything up to 60,000
print(reshape_train[image_index].shape)
train_img = reshape_train / 255
test_img = reshape_test / 255
print('x_train shape:', train_img.shape)
print('Number of images in train', train_img.shape[0])
print('Number of images in test', test_img.shape[0])
model, datagen = cnnmodel((28,28,1))
train_img_batch = datagen.flow(train_img, train_label, batch_size = 64)
test_img_batch = datagen.flow(test_img, test_label, batch_size = 64)
history = model.fit_generator(train_img_batch, epochs = 2)
score = model.evaluate(test_img, test_label)
print (score)
cnn_cam_mnist(model)

# img_idx = random.randint(1,len(test_img))
# print (img_idx)
# img = np.expand_dims(test_img[img_idx], axis=0)
# print(img.shape)
# preds = model.predict(img)
# pred = np.argmax(preds)
# print ('Prediction',pred)
# plt.imshow(test_img[img_idx].reshape(28,28), cmap='Greys')
# plt.show()


