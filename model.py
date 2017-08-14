# import keras lib
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import Adam
import preprocess
import csv
import numpy as np

lines = []
steer = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        steer.append(float(line[3]))



model = Sequential()
# cropping the image topPixel from top and botPixel from bottom
#model.add(Cropping2D(cropping=((70, 25), (4, 4)),input_shape=(160,320,3)))
'''# The Following structrue is LeNet 
First conv layer 6 filters with 5*5 kernel
model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

# Second conv layer 6 filters with 5*5 kernel
model.add(Convolution2D(16, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

# three fully connected layers
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))'''

# The Following structure is NVIDIA Model structure
model.add(Lambda(lambda x:(x/122.5) - 1, input_shape=(64,64,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))



model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))


model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.65))


model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.65))

model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# configure learning process 
from keras.optimizers import Adam
learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate))
# train the model with fixed number of epochs
bias_shreshold = 0.85
single_generator = preprocess.generator(lines,64,64,bias_shreshold)
valid_generator = preprocess.generator(lines,64,64,bias_shreshold)
test_generator = preprocess.generator(lines,64,64,bias_shreshold)
model.fit_generator(single_generator, samples_per_epoch=20032, nb_epoch=10, validation_data=valid_generator, nb_val_samples=3200)

#history_object = model.fit_generator(train_generator, samples_per_epoch=len(tra))
model.save('model.h5') 
print('model saved.')