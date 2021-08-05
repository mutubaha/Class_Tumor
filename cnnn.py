"""
Python3

Created on 2021

@author: bahattin

Classification of benign and malignant tumors from thyroid gland ultrasound images 
(http://cimalab.intec.co/applications/thyroid/index.php)  

"""
## import libraries

import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense 

# initialization
classifier = Sequential()

# step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. convolution layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 3 - Flattening
classifier.add(Flatten())

# step 4 - YSA
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# CNN and images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)

import numpy as np
import pandas as pd


test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)
pred[pred > .5] = 1
pred[pred <= .5] = 0

print('prediction passed')

## accuracy 

test_labels = []

for i in range(0,int(6)):
    test_labels.extend(np.array(test_set[i][1]))
    
print('test_labels')
print(test_labels)

file_names = test_set.filenames

result = pd.DataFrame()
result['file_names']= file_names
result['predictions'] = pred
result['test'] = test_labels   

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print (cm)



