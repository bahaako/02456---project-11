from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.misc

# Testing on full image
N = 4650 # Images in test set

test_model = load_model('CNN_model.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255) # Scale image values

generator = test_datagen.flow_from_directory(
        'data/test_frame',
        target_size=(150, 150),
        batch_size=1,
        class_mode=None,  # Import data without labels
        shuffle=False)

probabilities = test_model.predict_generator(generator, N)
y_pred = probabilities < 0.5

np.savetxt( 'data/test_frame/test_pred.txt', np.asarray(y_pred), fmt='%u' )
np.savetxt( 'data/test_frame/test_prob.txt', np.asarray(probabilities), fmt='%.4f' )
