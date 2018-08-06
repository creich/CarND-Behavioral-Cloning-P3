import csv
import cv2
import numpy as np


lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip header row
    next(reader)

    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    # append flipped versions of images as well
    #image_flipped = np.fliplr(image)
    #images.append(image_flipped)
    #measurement_flipped = -measurement
    #measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
#model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', input_shape=(160, 320, 3)))
model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(256))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
