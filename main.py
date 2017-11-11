import numpy as np
import os
#from keras.models import Sequential
#from keras.layers import Dense, Activation

import features


def main():
    np.savetxt('test.txt', features.ddmfcc('test.wav'), fmt='%.4f')
    # extract_feats('2003_nist_sre')
    # model = construct_model()


def extract_feats(directory):
    for root, dirs, files in os.walk(directory):
        wav_files = [file for file in files if file.endswith('.wav')]
        for file in wav_files:
            name = file.split('.')[0]
            np.save(os.path.join(root, name), features.mfcc(os.path.join(root, file)))
            print(os.path.join(root, name)+'.npy')


def construct_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model):
    with open('/2003_nist_sre/train_data/train_2sp/m_train.lst') as f:
        lines = f.readlines()


main()
