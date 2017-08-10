import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


lines = []
steer = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        steer.append(float(line[3]))
#plt.hist(steer)
# plt.show()

# change the size and normalize the image data
def change_size_and_normalize(image, new_row, new_col):
	image = cv2.resize(image, (new_col, new_row), interpolation=cv2.INTER_AREA)
	new_image = image/122.5 - 1
	return new_image

# cropping the image data to only focus on the path
def cropping(image, top_perc=0.40, bot_perc=0.12):
    top = int(np.ceil(image.shape[0]*top_perc))
    bottom = int(image.shape[0] - np.ceil(image.shape[0]*bot_perc))
    new_image = image[top:bottom,:]

    return new_image
    
 # flipping the image with a random probability
def flipping(image, angle, flip_prob=0.5):
    p = np.random.uniform()
    if p > 0.5:
        new_image = cv2.flip(image, 1)
        new_angle = angle*(-1.0)
        return new_image, new_angle
    else:
        return image, angle

# changing brightness
def brightness_change(image):
	temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	temp = np.array(temp, dtype = np.float64)
	brigh_rand = .25+ np.random.uniform()
	temp[:,:,2] = temp[:,:,2] * brigh_rand
	temp[:,:,2][temp[:,:,2]>255] = 255
	temp = np.array(temp, dtype=np.uint8)
	new_image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

	return new_image 

# change position of image and steering angle
def translation(image, angle, trans_range=100):
    rows, cols, ch = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range/2
    tr_y = 40 * np.random.uniform() - 40/2
    trans_mat = np.float32([[1,0,tr_x],[0,1,tr_y]])
    new_angle = angle + tr_x/trans_range*2*.2
    new_image = cv2.warpAffine(image, trans_mat, (cols, rows))

    return new_image, new_angle

# preprocessing procedure for a single image
def preprocess_img(line, new_row, new_col, correction):
    camera_pos = np.random.randint(3)
    angle = float(line[3])
    path = './IMG/' + line[camera_pos].split('\\')[-1]
    if camera_pos == 1:
        angle = angle + correction
    if camera_pos == 2:
        angle = angle - correction
    image = np.array(cv2.imread(path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cropping(image)

    image = brightness_change(image)

    image, angle = translation(image, angle, 100)

    image = change_size_and_normalize(image, new_row, new_col)
    
    image, angle = flipping(image, angle)
    
    return image, angle

# Generator 
def generator(lines, new_row, new_col, BATCH_SIZE = 64):
    batch_img = np.zeros((BATCH_SIZE, new_row, new_col, 3))
    batch_ang = np.zeros(BATCH_SIZE)
    bias_threshold_1 = 0.9
    bias_threshold_2 = 0.6
    angle_threshold_1 = 0.1
    angle_threshold_2 = 0.7
    # infinite loop for generating data batches
    while 1:
        for i in range(BATCH_SIZE):
            keep_pro = 0
            while keep_pro ==0:
                line = lines[np.random.randint(len(lines))]
                temp_img, temp_ang = preprocess_img(line, new_row,new_col,0.229)
        
                if abs(temp_ang) < angle_threshold_1:
                    rand_prob = np.random.uniform()
                       # print(temp_ang, rand_prob)
                    if rand_prob > bias_threshold_1:
                        batch_img[i], batch_ang[i] = temp_img, temp_ang
                        keep_pro = 1
                elif abs(temp_ang)<angle_threshold_2:
                    rand_prob = np.random.uniform()
            # print(temp_ang, rand_prob)
                    if rand_prob > bias_threshold_2:
                        batch_img[i], batch_ang[i] = temp_img, temp_ang
                        keep_pro = 1
                else:
                    batch_img[i], batch_ang[i] = temp_img, temp_ang
                    keep_pro = 1

        #print(batch_img, batch_ang)
        yield (batch_img, batch_ang)
