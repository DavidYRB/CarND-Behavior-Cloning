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

# change the size of the image
def change_size_and normalize(image, new_row, new_col):
	image = cv2.resize(image, (new_col, new_row), interpolation=cv2.INTER_AREA)
	new_image = image/122.5 - 1
	return new_image

# flipping
def flipping(image, angle):
	new_image = cv2.flip(image, 1)
	new_angle = angle*(-1.0)

	return new_image, new_angle

# changing brightness
def brightness_change(image):
	temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	temp = np.array(temp, dtype = np.float64)
	brigh_rand = .25+ np.random.uniform()
	temp[:,:,2] = temp[:,:,2] * brigh_rand
	temp[:,:,2][temp[:,:,2]>255] = 255
	temp = np.array(temp, dtype=np.unit8)
	new_image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

	return new_image 

# translation of image along x and y axis
def translation(image, angle, trans_range):
	rows, cols, ch = image.shape
	tr_x = trans_range * np.random.uniform() - trans_range/2
 	tr_y = trans_range * np.random.uniform() - trans_range/2
 	trans_mat = np.float32([[1,0,tr_x],[0,1,tr_y]])
 	new_angle = angle + tr_x/trans_range*2*.2
 	new_image = cv2.warpAffine(image, trans_mat, (cols, rows))

 	return new_image, new_angle

 # add shadows 
 def add_shadow(image):
 	top_y = 320*np.random.uniform()
 	bot_y = 320*np.random.uniform()
 	top_x = 0
 	bot_x = 160
 	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
 	shadow_mask = 0*image_hls[:,:,1]
 	X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
 	Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) - (bot_x-top_x)*(Y_m-top_y) >=0)] = 1
    if np.random.randint(2) == 1:
    	random_bright = .5
    	cond1 = shadow_mask == 1
    	cond0 = shadow_mask == 0
    	if np.random.randint(2) == 1:
    		image_hls[:,:,1][cond1] == image_hls[:,:,1][cond1]*random_bright
    	else:
            image_hls[:,:,1][cond0] == image_hls[:,:,1][cond0]*random_bright
    new_image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB) 
    
    return new_image   		

# preprocessing procedure for a single image
def preprocess_img(line, correction):
	camera_pos = np.random.randint(3)
	angle = line[camera_pos]
	path = './IMG/' + line[camera_pos].split('\\')[-1]
	if camera_pos == 1:
		angle = angle + correction
	if camera_pos == 2:
		angle = angle - correction
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, angle = translation(image, angle, 100)
    image = brightness_change(image)
    image = add_shadow(image)

    if np.random.randint(2) == 1:
    	new_image, new_angle = flipping(image, angle)

    return new_image, new_angle		 


def generator(lines, new_row, new_col, BATCH_SIZE = 128):
	batch_img = np.zeros((BATCH_SIZE, new_row, new_col, 3))
	batch_ang = np.zeros(BATCH_SIZE)
	# infinite loop for generating data batches
	while 1:
		for i in range(BATCH_SIZE):
			batch_img[i] = preprocess_img() 

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