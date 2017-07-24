import csv
from skimage import io
from skimage.color import rgb2gray
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D,Cropping2D,Lambda


current_path = 'data/IMG/'
delim = '/'

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            correction = 0.2 # this is a parameter to tune
            for batch_sample in batch_samples:
                image_center_path = batch_sample[0]
                image_left_path = batch_sample[1]
                image_right_path = batch_sample[2]
                
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
              
                image_center = io.imread(current_path + image_center_path.split(delim)[-1])
                image_left = io.imread(current_path + image_left_path.split(delim)[-1])
                image_right = io.imread(current_path + image_right_path.split(delim)[-1])
                
                images.extend([image_center, image_left, image_right])
                measurements.extend([steering_center, steering_left, steering_right])
                
                images.extend([np.fliplr(image_center), np.fliplr(image_left), np.fliplr(image_right)])
                measurements.extend([-steering_center, -steering_left, -steering_right])
                              
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
                        
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


def lenet():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(6, (5, 5),activation='relu'))
    model.add(MaxPooling2D())       
    
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())        
    
    model.add(Flatten())
    model.add(Dense(120))    
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


batch_size = 32

model = lenet()

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_samples)//batch_size, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)//batch_size,
                    epochs=20)


model.save('model_1.h5')