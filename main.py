from keras.models import load_model
model = load_model('weights.best.basic_mlp.hdf5')

import numpy as np
# from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.optimizers import Adam
# from keras.utils import np_utils
# from sklearn import metrics
import librosa
# import librosa.display
import os
le = LabelEncoder()
dataset_classes=[]
with open('classes.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        dataset_classes.append(currentPlace)
dataset_classes = np.array(dataset_classes)
le.fit_transform(dataset_classes)


def extract_feature(file_name):

    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file")
        return None, None

    return np.array([mfccsscaled])

def print_prediction(file_name):
    prediction_feature = extract_feature(file_name)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )

filename='typing.wav'
print_prediction(filename)
