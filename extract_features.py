import numpy as np
import os
import soundfile as sf
import python_speech_features


def extract_features(directory, nfeats):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                with open(os.path.join(root, file), 'rb') as f:
                    data, samplerate = sf.read(f)
                    feats = python_speech_features.logfbank(data, samplerate=samplerate, nfilt=nfeats, winfunc=np.hamming)
                    inputs = []
                    with open(os.path.join(root, file.split('.')[0]) + '.lab_ph') as g:
                        phones = g.readlines()
                        for phone in phones:
                            start = int(float(phone.split()[0]) * 100)
                            end = int(float(phone.split()[0]) * 100)
                            inputs.append(feats[start: end])
                    feat_file = os.path.join(root, file.split('.')[0])
                    np.save(feat_file, np.array(inputs))
                    print(feat_file)
