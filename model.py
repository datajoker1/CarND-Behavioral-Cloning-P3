import csv
import cv2
import numpy as np

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)



images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        image = cv2.imread(current_path)

        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

argumented_images, argmented_measurements = [], []
for image, measurement in zip(images, measurements):
    argumented_images.append(image)
    argmented_measurements.append(measurement)
    argumented_images.append(cv2.flip(image, 1))
    argmented_measurements.append(measurement * -1.0)

X_train = np.array(argumented_images)
y_train = np.array(argmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout

model = Sequential()
model.add(Lambda(lambda x:x/127.5 - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, 5, strides=(2,2) ,activation='relu'))
model.add(Conv2D(36, 5, strides=(2,2) ,activation='relu'))
model.add(Conv2D(48, 5, strides=(2,2) ,activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, shuffle=True, nb_epoch=5)

model.save('model.h5')