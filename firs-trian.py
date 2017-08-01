import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Define Augmentation functinos

# flipping
def flipping(image, angle):
	new_image = cv2.flip(image, 1)
	new_angle = angle*(-1.0)

	return new_image, new_angle

# changing brightness
def brightness_change(image):
	temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	temp = np.array(temp, dtype = np.float64)
	brigh_rand = .25+ np.random.uniform()
	temp[:,:,2] = temp[:,:,2] * brigh_rand
	temp[:,:,2][temp[:,:,2]>255] = 255
	temp = np.array(temp, dtype=np.unit8)
	image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

	return image 


images = []
measurements = []
correction = 0.1
for line in lines:
	filenames = []
	imagesPath = []
	for i in range(3):
		filename =  line[i].split('\\')[-1]
		filenames.append(filename)
		current_path = './IMG/' + filename
		imagesPath.append(current_path)
	# read images from imagesPath	
	imageCenter = cv2.imread(imagesPath[0])
	imageLeft = cv2.imread(imagesPath[1])
	imageRight = cv2.imread(imagesPath[2])

	measureCenter = float(line[3]) 
	measureLeft = float(line[3]) + correction
	measureRight = float(line[3]) - correction
	images.extend([imageCenter, imageLeft, imageRight])
	measurements.extend([measureCenter, measureLeft, measureRight])

# Augment data set by flipping all images
X_augment, y_augment = [], []
for x, y in zip(images, measurements):
	X_augment.append(x)
	y_augment.append(y)
	X_augment.append(cv2.flip(x,1))
	y_augment.append(y*(-1.0)) 

# turn all data into array 
X_train = np.array(X_augment)
y_train = np.array(y_augment)

# import keras lib
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



model = Sequential()
# cropping the image topPixel from top and botPixel from bottom
topPixel = 70
botPixel = 25
model.add(Cropping2D(cropping=((topPixel, botPixel),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/122.5 -1))

# The Following structrue is LeNet 
# First conv layer 6 filters with 5*5 kernel
# model.add(Convolution2D(6, 5, 5, border_mode='valid', activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

# # Second conv layer 6 filters with 5*5 kernel
# model.add(Convolution2D(16, 5, 5, border_mode='valid', activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

# # three fully connected layers
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(1))

# The Following structure is NVIDIA Model structure
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.65))

model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.65))

model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.65))

model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.65))

model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='valid', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.65))

model.add(Flatten())
model.add(Dense(1164))
model.add(ELU())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

# configure learning process 
model.compile(loss='mse', optimizer='adam')
# train the model with fixed number of epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, show_accuracy=True)

# visualize the training loss and validation loss
#history_object = model.fit_generator(train_generator, samples_per_epoch=len(tra))
model.save('model.h5') 