
# coding: utf-8

# In[1]:


'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12




def main(unused_argv):

    f = open('train_files_test.csv', 'r')
    g = open('test_files_test.csv', 'r')
    train_data = []
    train_labels = []
    #  lines = f.readline()

    #  lines = lines.strip()
    # columns = lines.split(',')
    #fname = columns[0]
    # labels = columns[1]
    # os.chdir('train') 

    # im=tf.read_file(fname)
    # image = tf.image.decode_image(im)
    # image=tf.cast(image,tf.int32)
    # image = tf.image.resize_nearest_neighbor(image,[750,750])
    # train_data=image
    # train_labels = labels
    #os.chdir('..')



    ##################Training#####################
    for line in f:
        line = line.strip()
        columns = line.split(',')
        fname = columns[0]
        label = columns[1]
        os.chdir('train') 

        im=tf.read_file(fname)
        image = tf.image.decode_image(im)
        image=tf.cast(image,tf.int32)
        image = tf.image.resize_nearest_neighbor(image,[750,750])
        train_data.append(image)
        train_labels.append(label)


    #may need to conver back to tensor
        os.chdir('..')

    f.close()
    train_data=tf.stack(train_data)
    train_labels=tf.stack(train_labels)

    g = open('test_files_test.csv', 'r')

    #  lines = g.readline()

    #  lines = lines.strip()
    #  columns = lines.split(',')
    #  fname = columns[0]
    #  labels = columns[1]
    #  os.chdir('test') 

    # im=tf.read_file(fname)
    # image = tf.image.decode_image(im)
    # image = tf.image.resize_nearest_neighbor(image,[750,750])
    # eval_data=image
    # eval_labels = labels
    # os.chdir('..')

    eval_data = []
    eval_labels = []

    for line in g:
        line = line.strip()
        columns = line.split(',')
        fname = columns[0]
        label = columns[1]
        os.chdir('test') 

        im=tf.read_file(fname)
        image = tf.image.decode_image(im)
        image=tf.cast(image, tf.int32)
        image = tf.image.resize_nearest_neighbor(image,[750,750])
        #    eval_data=tf.concat(eval_data,image,0)
        eval_data.append(image)
        eval_labels.append(label)


        os.chdir('..')

    g.close()    

    eval_data=tf.stack(eval_data)
    eval_labels=tf.stack(eval_labels)

    x_train=train_data
    y_train=train_labels

    x_test=eval_data
    y_test=eval_labels


    # input image dimensions
    img_rows, img_cols = 750, 750

    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #THIS NEEDS TO BE CHANGED
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

