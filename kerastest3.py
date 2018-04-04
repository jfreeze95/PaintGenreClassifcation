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
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
from sklearn.preprocessing import LabelBinarizer

batch_size = 1
num_classes = 18
epochs = 12




def main(unused_argv):
    ###########Make Vector of Possible Classes#############
    genres = ["portrait","genre painting", "landscaping", "abstract", "religious painting", "cityscape", "sketch and study", "illustration", "still life", "symbolic painting", "figurative","nude painting (nu)", "mythological painting", "design", "marina", "flower paiunting", "animal painting", "self-portrait"]
    encoder = LabelBinarizer()
    bin_labels = encoder.fit_transform(genres)
    print(bin_labels)

    ###########Bringing in Files ##################
    #sess = tf.InteractiveSession()
    f = open('train_files.csv', 'r')
    print("1")
    g = open('test_files.csv', 'r')
    train_data = []
    train_labels = []
    print("2")



    ##################Training#####################
    count = 0
    for line in f:
    	#print ('in train')
        line = line.strip()
        columns = line.split(',')
        fname = columns[0]
        #print(fname)
        label = columns[1]
        os.chdir('train')
        #print(os.system('pwd'))
        #print(os.system('find . -name {}'.format(fname)))
        tf.convert_to_tensor(fname, dtype=tf.string)
        filename_queue = tf.train.string_input_producer([fname])
        reader = tf.WholeFileReader()
        key, im = reader.read(filename_queue)
        #im=tf.read_file(str(fname))
        image = tf.image.decode_image(im)
        image=tf.cast(image,tf.int32)
        image = tf.image.resize_nearest_neighbor(image,[750,750])
        train_data.append(image)
        train_labels.append(label)
        count += 1
        os.chdir('..')
        if count > 10: 
            break
        
    
    train_data=tf.stack(train_data)
    train_labels=tf.stack(train_labels)

    g = open('test_files.csv', 'r')


    eval_data = []
    eval_labels = []

    count2 = 0
    for line in g:
    	#print ('in test')
        line = line.strip()
        columns = line.split(',')
        fname = columns[0]
        label = columns[1]
        os.chdir('test') 
        #print(os.system('pwd'))
        #print(os.system('find . -name {}'.format(fname)))
        filename_queue = tf.train.string_input_producer([fname])
        reader = tf.WholeFileReader()
        key, im = reader.read(filename_queue)
        image = tf.image.decode_image(im)
        image=tf.cast(image, tf.int32)
        image = tf.image.resize_nearest_neighbor(image,[750,750])
        #    eval_data=tf.concat(eval_data,image,0)
        eval_data.append(image)
        eval_labels.append(label)
        os.chdir('..')
        count2 += 1
        if count2 >10: 
            break


    #print ('out of test loop')   

    eval_data=tf.stack(eval_data)
    eval_labels=tf.stack(eval_labels)
    #print ('evall stacked')

    x_train=train_data
    y_train=train_labels
    
    x_test=eval_data
    y_test=eval_labels


    # input image dimensions
    img_rows, img_cols = 750, 750

    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = tf.reshape(x_train, [x_train.shape[0], 3, img_rows, img_cols])
        
        x_test = tf.reshape(x_test, [x_test.shape[0], 3, img_rows, img_cols])
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = tf.reshape(x_train, [x_train.shape[0], img_rows, img_cols, 3])
        x_test = tf.reshape(x_test, [x_test.shape[0], img_rows, img_cols, 3])
        input_shape = (img_rows, img_cols, 3)

    #print ('reshaped')
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test,dtype=tf.float32)
    
    sess = tf.InteractiveSession()
    y_train = sess.run(y_train)
    print('x trained')
    y_test = sess.run(y_test)
    print ('x tested')
    #with tf.Session() as sess:
     
    
    print(type(x_train))
    #THIS NEEDS TO BE CHANGED
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    print ('y_train')
    print (y_train)
    y_train = encoder.transform(y_train)
    print(y_train)
    y_test = encoder.transform(y_test)
    print(type(y_test))

    #include get's infinite wait time, not include get ndim error
    #x_test = tf.Session().run(x_test)


   
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)
    print("1 potato")
    model = Sequential()
    print("2 potato")
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    print("3 potato")
    model.add(Conv2D(64, (3, 3), activation='relu'))
    print("4 potato")
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    print("5 potato")
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print("6 potato")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print("7 potato")
    score = model.evaluate(x_test, y_test, verbose=0)
    print("more")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    f.close()
    g.close()
    
if __name__ == "__main__":
  tf.app.run()



