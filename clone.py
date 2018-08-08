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

steering_correction = 0.09

print("loading training data...")

for line in lines:
   # use images from left and right camera as well
   for index in range(1):
      source_path = line[index]
      filename = source_path.split('/')[-1]
      current_path = '../data/IMG/' + filename
      image = cv2.imread(current_path)
      #image = cv2.resize(image, dsize=(32,32), interpolation = cv2.INTER_CUBIC)

      image = np.asarray(image)
      # convert BGR to RGB
      image = image[:,:,::-1]

      images.append(image)
      measurement = float(line[3])

      # manipulation of steering angle to use left and right pictures as well
      if index == 1:
         measuremnt = measurement + steering_correction
      elif index == 2:
         measuremnt = measurement - steering_correction

      measurements.append(measurement)

      # append flipped versions of images as well
      image_flipped = np.fliplr(image)
      images.append(image_flipped)
      measurement_flipped = -measurement
      measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)

print("done.")

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D

dropout_rate = 0.23

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', input_shape=(160, 320, 3)))
model.add(Conv2D(6, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))
model.add(Conv2D(12, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))
model.add(Conv2D(18, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
