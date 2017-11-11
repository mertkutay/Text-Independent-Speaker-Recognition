import numpy as np
import os

from network import Network
import features


def main():
    extract_feats('2003_nist_sre')


def extract_feats(directory):
    for root, dirs, files in os.walk(directory):
        wav_files = [file for file in files if file.endswith('.wav')]
        for file in wav_files:
            name = file.split('.')[0]
            np.save(os.path.join(root, name), features.ddmfcc(os.path.join(root, file), 12))


main()
