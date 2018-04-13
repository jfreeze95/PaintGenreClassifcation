#!/usr/bin/python

# coding: utf-8

# In[3]:

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from sklearn.preprocessing import LabelBinarizer

img_width = 750
img_height = 750
batch_size = 1
num_classes = 18
epochs = 1
nb_train_samples = 3
nb_validation_samples = 4
#we have 68087 train images
#we have 20342 test images

def main(unused_argv):
	train_data_dir = 'foolingaround/trainfiles'
	test_data_dir = 'foolingaround/testfiles'

	#can alter these paramemters and many more not included to alter how pictures are read in.
	train_datagen = ImageDataGenerator(
		horizontal_flip=False)

	test_datagen = ImageDataGenerator(
		horizontal_flip=False)


	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size = (img_width,img_height),
		batch_size = batch_size,
		class_mode ='categorical')

	#print('train_generator')
	#print(train_generator[0])

	validation_generator = test_datagen.flow_from_directory(
		test_data_dir,
		target_size = (img_width,img_height),
		batch_size =batch_size,
		class_mode ='categorical')

	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width,img_height)
	else:
		input_shape = (img_width,img_height, 3)



	#y_train = keras.utils.to_categorical(y_train, num_classes)
	#y_test = keras.utils.to_categorical(y_test, num_classes)
	print("1 potato")
	model = Sequential()
	print("2 potato")
	model.add(Conv2D(32, kernel_size=(3, 3),
		activation='relu',
		input_shape=input_shape))
	print("3 potato")
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	print("4 potato")
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	#model.add(Dropout(0.5))
	print("5 potato")
	model.add(Dense(18, activation='softmax'))

	model.compile(loss=keras.losses.binary_crossentropy,
		optimizer=keras.optimizers.Adam(),
		metrics=['accuracy'])
	print('6 potato')
	model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size)
	print("7 potato")
	score = model.evaluate_generator(validation_generator)
	print("predictions")
	predictions=model.predict_generator(validation_generator)
	print(predictions)
	print("more")
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


if __name__ == "__main__":
	tf.app.run()
