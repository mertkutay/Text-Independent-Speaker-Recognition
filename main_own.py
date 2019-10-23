import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt, ticker
from network import Network

import features


EPOCHS = 5
FRAME_SIZE = 13
NUM_FRAMES = 9
INPUT_SIZE = 2 * FRAME_SIZE * NUM_FRAMES
BATCH_SIZE = 32
DATA_SIZE = 1000000


def main():
    # np.savetxt('5483.txt', features.mfcc('5483.wav'), fmt='%.4f')
    # extract_feats('2003_nist_sre')
    model = construct_model()
    model = train_model(model)
    # model.save('model.h5')
    # model = load_model('model.h5')
    eer = test_model_utterance(model)
    print('EER Accuracy: {}'.format(eer))


def extract_feats(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        wav_files.extend([os.path.join(root, file) for file in files if file.endswith('.wav')])
    for file in wav_files:
        feats = features.mfcc(file)
        if feats is not None:
            np.save(file.split('.')[0], feats)
        # else:
        #     os.remove(file.split('.')[0] + '.npy')


def construct_model():
    model = Network([INPUT_SIZE, INPUT_SIZE, 100, 1])
    return model


def train_model(model):
    npy_files = []
    for root, dirs, files in os.walk('2003_nist_sre/train_data'):
        npy_files.extend([os.path.join(root, file) for file in files if file.endswith('.npy')])
    data = np.empty((DATA_SIZE, 2 * NUM_FRAMES, FRAME_SIZE))
    target = np.empty((DATA_SIZE, 1))
    index = 0
    for file in npy_files:
        first = np.load(file)
        other1 = np.random.choice(npy_files)
        while other1 == file:
            other1 = np.random.choice(npy_files)
        second = np.load(other1)
        # other2 = np.random.choice(npy_files)
        # while other2 == file:
        #     other2 = np.random.choice(npy_files)
        # third = np.load(other2)
        # other3 = np.random.choice(npy_files)
        # while other3 == file:
        #     other3 = np.random.choice(npy_files)
        # fourth = np.load(other3)
        len1 = len(first) - NUM_FRAMES + 1
        len2 = len(second) - NUM_FRAMES + 1
        # len3 = len(third) - NUM_FRAMES + 1
        # len4 = len(fourth) - NUM_FRAMES + 1
        for i in range(0, len1, NUM_FRAMES-3):
            data[index, : NUM_FRAMES] = first[i: i + NUM_FRAMES]
            m = np.random.choice(len1)
            while m == i:
                m = np.random.choice(len1)
            data[index, NUM_FRAMES: 2 * NUM_FRAMES] = first[m: m + NUM_FRAMES]
            target[index] = 1
            data[index + 1, : NUM_FRAMES] = first[i: i + NUM_FRAMES]
            n = np.random.choice(len2)
            data[index + 1, NUM_FRAMES: 2 * NUM_FRAMES] = second[n: n + NUM_FRAMES]
            target[index + 1] = 0
            # data[index + 2, : NUM_FRAMES] = first[i: i + NUM_FRAMES]
            # n = np.random.choice(len3)
            # data[index + 2, NUM_FRAMES: 2 * NUM_FRAMES] = third[n: n + NUM_FRAMES]
            # target[index + 2] = 0
            # data[index + 3, : NUM_FRAMES] = first[i: i + NUM_FRAMES]
            # n = np.random.choice(len4)
            # data[index + 3, NUM_FRAMES: 2 * NUM_FRAMES] = fourth[n: n + NUM_FRAMES]
            # target[index + 3] = 0
            index += 2
            if index + 2 > DATA_SIZE:
                data = data[:index]
                target = target[:index]
                model.train(list(zip(np.reshape(data, (-1, INPUT_SIZE, 1)), target)), epochs=1, batch_size=BATCH_SIZE, learning=0.5, regularization=0.01)
                data = np.empty((DATA_SIZE, 2 * NUM_FRAMES, FRAME_SIZE))
                target = np.empty((DATA_SIZE, 1))
                index = 0
    data = data[:index]
    target = target[:index]
    model.train(list(zip(np.reshape(data, (-1, INPUT_SIZE, 1)), target)), epochs=EPOCHS, batch_size=BATCH_SIZE, learning=5, regularization=1)
    return model


def test_model_batch(model):
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2003_nist_sre/test_data/answer_keys/1sp-lim/KEY_1sp.v10')
    with open(key_file) as f:
        lines = f.readlines()
    test_list = [[line.split()[0], line.split()[3], line.split()[4]] for line in lines]
    y_pred = np.empty((30000000, 1))
    y_true = np.empty_like(y_pred)
    index = 0
    for k in range(len(test_list)):
        auth_file = test_list[k][0]
        acc_file = test_list[k][1]
        imp_file = test_list[np.random.choice(len(test_list))]
        while imp_file[1] == acc_file:
            imp_file = test_list[np.random.choice(len(test_list))]
        acc_gender = '_1sp_male/' if test_list[k][2] == 'm' else '_1sp_female/'
        imp_gender = '_1sp_male/' if imp_file[2] == 'm' else '_1sp_female/'
        try:
            auth = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '2003_nist_sre/test_data/test' + acc_gender + auth_file + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        try:
            acc = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '2003_nist_sre/train_data/train' + acc_gender + acc_file + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        try:
            imp = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '2003_nist_sre/train_data/train' + imp_gender + imp_file[1] + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        len1 = int(len(auth)) - NUM_FRAMES + 1
        len2 = int(len(acc)) - NUM_FRAMES + 1
        len3 = int(len(imp)) - NUM_FRAMES + 1
        if len1 <= 1 or len2 <= 1 or len3 <= 1:
            print(auth_file + ' ' + acc_file + ' ' + imp_file[1])
            continue
        data = np.empty((2 * len1, 2 * NUM_FRAMES, FRAME_SIZE))
        for i in range(0, len1):
            data[i, : NUM_FRAMES] = auth[i: i + NUM_FRAMES]
            m = np.random.choice(len2)
            data[i, NUM_FRAMES: 2 * NUM_FRAMES] = acc[m: m + NUM_FRAMES]
            data[len1 + i, : NUM_FRAMES] = auth[i: i + NUM_FRAMES]
            n = np.random.choice(len3)
            data[len1 + i, NUM_FRAMES: 2 * NUM_FRAMES] = imp[n: n + NUM_FRAMES]
        predictions = model.predict(np.reshape(data, (-1, INPUT_SIZE)))
        y_true[index:index+len1] = 1
        y_true[index+len1:index+2*len1] = 0
        y_pred[index:index+len1] = predictions[:len1]
        y_pred[index+len1:index+2*len1] = predictions[len1:]
        index += 2*len1
    y_true = y_true[:index]
    y_pred = y_pred[:index]
    eer = test_metrics(y_true, y_pred)
    return eer


def test_model_utterance(model):
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2003_nist_sre/test_data/answer_keys/1sp-lim/KEY_1sp.v10')
    with open(key_file) as f:
        lines = f.readlines()
    test_list = [[line.split()[0], line.split()[3], line.split()[4]] for line in lines]
    y_pred = np.empty((2 * len(test_list), 1))
    y_true = np.empty_like(y_pred)
    index = 0
    for k in range(len(test_list)):
        auth_file = test_list[k][0]
        acc_file = test_list[k][1]
        imp_file = test_list[np.random.choice(len(test_list))]
        while imp_file[1] == acc_file:
            imp_file = test_list[np.random.choice(len(test_list))]
        acc_gender = '_1sp_male/' if test_list[k][2] == 'm' else '_1sp_female/'
        imp_gender = '_1sp_male/' if imp_file[2] == 'm' else '_1sp_female/'
        try:
            auth = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '2003_nist_sre/test_data/test' + acc_gender + auth_file + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        try:
            acc = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '2003_nist_sre/train_data/train' + acc_gender + acc_file + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        try:
            imp = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '2003_nist_sre/train_data/train' + imp_gender + imp_file[1] + '.npy'))
        except(FileNotFoundError, IOError):
            continue
        len1 = int(len(auth)) - NUM_FRAMES + 1
        len2 = int(len(acc)) - NUM_FRAMES + 1
        len3 = int(len(imp)) - NUM_FRAMES + 1
        if len1 <= 0 or len2 <= 0 or len3 <= 0:
            print(auth_file + ' ' + acc_file + ' ' + imp_file[1])
            continue
        y_true[index] = 1
        y_true[index + 1] = 0
        data = np.empty((2 * len1, 2 * NUM_FRAMES, FRAME_SIZE))
        for i in range(0, len1):
            data[i, : NUM_FRAMES] = auth[i: i + NUM_FRAMES]
            m = np.random.choice(len2)
            data[i, NUM_FRAMES: 2 * NUM_FRAMES] = acc[m: m + NUM_FRAMES]
            data[len1 + i, : NUM_FRAMES] = auth[i: i + NUM_FRAMES]
            n = np.random.choice(len3)
            data[len1 + i, NUM_FRAMES: 2 * NUM_FRAMES] = imp[n: n + NUM_FRAMES]
        predictions = model.predict(np.reshape(data, (-1, INPUT_SIZE)))
        y_pred[index] = np.mean(predictions[:len1])
        y_pred[index + 1] = np.mean(predictions[len1:])
        index += 2
    y_true = y_true[:index]
    y_pred = y_pred[:index]
    eer = test_metrics(y_true, y_pred)
    return eer


def test_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    y_dec = np.zeros_like(y_true)
    y_dec[y_pred > thresholds[idx]] = 1
    write_confusion(y_true, y_dec)
    plot_roc(fpr, tpr)
    plot_det(fpr, fnr, idx)
    eer = np.mean(y_true != y_dec)
    return eer


def write_confusion(y_true, y_dec):
    confusion = confusion_matrix(y_true, y_dec)
    np.savetxt('confusion.txt', confusion, fmt='%d')


def plot_roc(fpr, tpr):
    plt.figure()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc(fpr, tpr))
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_curve.png')


def plot_det(fpr, fnr, idx):
    fpr *= 100
    fnr *= 100
    fig, ax = plt.subplots()
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.plot(fpr, fnr, 'b', label='EER = %%%0.2f' % fpr[idx])
    plt.yscale('log')
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.axis([1, 100, 1, 100])
    plt.legend(loc='lower left')
    plt.plot(fpr[idx], fnr[idx], 'rx')
    plt.ylabel('False Negative Rate (%)')
    plt.xlabel('False Positive Rate (%)')
    plt.savefig('det_curve.png')


main()
