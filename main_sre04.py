import numpy as np
import os
import time
from multiprocessing import Pool
import random
import soundfile as sf
import python_speech_features
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt, ticker
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, MaxoutDense, Activation, Input
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger


EPOCHS = 500
NUM_FEATS = 40
NUM_FRAMES = 20
INPUT_SIZE = NUM_FEATS * NUM_FRAMES
HIDDEN_SIZE = 256
NUM_SPEAKERS = 256
BATCH_SIZE = 128
DATA_SIZE = 100000
DATA_DIR = 'data\\sre04'
SPEAKER_DATA_DIR = os.path.join(DATA_DIR, 'speaker_data')
VECTOR_DATA_DIR = os.path.join(DATA_DIR, 'vector_data')
METRIC_DATA_DIR = os.path.join(DATA_DIR, 'metric_data')
NUM_STEPS = 2


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
                            end = int(float(phone.split()[1]) * 100)
                            inputs.append(feats[start: end])
                    inputs = np.concatenate(inputs, axis=0)
                    inputs = inputs[:len(inputs) - len(inputs) % NUM_FRAMES]
                    inputs = np.reshape(inputs, (-1, INPUT_SIZE))
                    feat_file = os.path.join(root, file.split('.')[0])
                    np.save(feat_file, inputs)
                    print(feat_file)


def write_speaker_data(files):
    speaker_data = []
    for j, file in enumerate(files[1]):
        speaker_data.append(np.load(file + '.npy'))
    speaker_data = np.concatenate(speaker_data, axis=0)
    np.save(os.path.join(SPEAKER_DATA_DIR, '{}'.format(files[0])), speaker_data)


def create_speaker_files():
    with open(os.path.join(DATA_DIR, 'spk2utt')) as f:
        lines = f.readlines()

    all_files = []
    for i, line in enumerate(lines):
        tokens = line.split()
        all_files.append((tokens[0], [os.path.join(DATA_DIR, token.split('-')[2]) for token in tokens[1:] if token.split('-')[2].startswith('t')]))

    pool = Pool()
    pool.map(write_speaker_data, iter(all_files))


def create_train_valid_data():
    order = 0
    step_size = 2500

    files = [os.path.join(SPEAKER_DATA_DIR, name) for name in os.listdir(SPEAKER_DATA_DIR)]
    files.sort(key=lambda file: os.path.getsize(file), reverse=True)
    files = files[:NUM_SPEAKERS]

    num_samples = np.load(files[-1]).shape[0]
    last_step = num_samples % step_size
    global NUM_STEPS
    NUM_STEPS = int(num_samples / step_size)

    for i in range(NUM_STEPS + 1):
        x_train = np.empty((step_size * NUM_SPEAKERS, INPUT_SIZE))
        y_train = np.zeros((step_size * NUM_SPEAKERS, NUM_SPEAKERS))
        count = 0
        for speaker, file in enumerate(files):
            feats = np.load(file)
            if i < NUM_STEPS:
                x_train[count: count + step_size, :] = feats[order: order + step_size, :]
                y_train[count: count + step_size, speaker] = 1
                count += step_size
            else:
                x_train[count: count + last_step, :] = feats[order: order + last_step, :]
                y_train[count: count + last_step, speaker] = 1
                count += last_step
        order += step_size

        if i == NUM_STEPS:
            x_train = x_train[:count]
            y_train = y_train[:count]
            np.save(os.path.join(VECTOR_DATA_DIR, 'x_valid'), x_train)
            np.save(os.path.join(VECTOR_DATA_DIR, 'y_valid'), y_train)
            return
        np.save(os.path.join(VECTOR_DATA_DIR, 'x_train{}'.format(i)), x_train)
        np.save(os.path.join(VECTOR_DATA_DIR, 'y_train{}'.format(i)), y_train)


def get_train_data(step):
    x_train = np.load(os.path.join(VECTOR_DATA_DIR, 'x_train{}.npy'.format(step)))
    y_train = np.load(os.path.join(VECTOR_DATA_DIR, 'y_train{}.npy'.format(step)))
    return x_train, y_train


def get_valid_data():
    x_valid = np.load(os.path.join(VECTOR_DATA_DIR, 'x_valid.npy'))
    y_valid = np.load(os.path.join(VECTOR_DATA_DIR, 'y_valid.npy'))
    return x_valid, y_valid


def train_vector_model():
    model = Sequential()

    model.add(Dense(HIDDEN_SIZE, input_shape=(INPUT_SIZE, )))
    model.add(Activation('relu'))

    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation('relu'))

    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(HIDDEN_SIZE))

    model_input = Input(shape=(INPUT_SIZE, ))
    features = model(model_input)

    extractor = Model(inputs=model_input, outputs=features)

    last = Activation('relu')(features)
    last = Dropout(0.5)(last)
    last = Dense(NUM_SPEAKERS, activation='softmax')(last)

    trainer = Model(inputs=model_input, outputs=last)

    sgd = SGD(lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=4, min_lr=0.0000001)
    csv_logger = CSVLogger('training_vector.log')

    trainer.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    x_valid, y_valid = get_valid_data()

    # for i in range(EPOCHS):
    #     print('{}/{} epochs'.format(i, EPOCHS))
    #     for j in range(NUM_STEPS):
    #         x_train, y_train = get_train_data(j)
    #         trainer.fit(x_train, y_train, epochs=i*NUM_STEPS+j+1, verbose=2, initial_epoch=i*NUM_STEPS+j, validation_data=(x_valid, y_valid), callbacks=[early_stopping, reduce_lr, csv_logger])

    x_train, y_train = get_train_data(0)
    trainer.fit(x_train, y_train, epochs=EPOCHS, verbose=2, validation_data=(x_valid, y_valid), callbacks=[early_stopping, reduce_lr, csv_logger])

    return extractor


def match_couples():
    speakers = [os.path.join(SPEAKER_DATA_DIR, name) for name in os.listdir(SPEAKER_DATA_DIR)]
    speakers.sort(key=lambda file: os.path.getsize(file), reverse=True)
    speakers = [os.path.basename(speaker).split('.')[0] for speaker in speakers]
    train_speakers = speakers[:NUM_SPEAKERS]
    test_speakers = speakers[NUM_SPEAKERS:]

    with open(os.path.join(DATA_DIR, 'spk2utt')) as f:
        lines = f.readlines()

    train_files = []
    for speaker in train_speakers:
        line = [l for l in lines if l.startswith(speaker)]
        if len(line) > 0:
            tokens = line[0].split()
            train_files.extend(([(speaker, token.split('-')[2]) for token in tokens[1:] if token.split('-')[2].startswith('t')]))

    with open(os.path.join(DATA_DIR, 'train_matches.txt'), 'w') as g:
        for i in range(len(train_files) - 1):
            j = i + 1
            while train_files[i][0] == train_files[j][0]:
                g.write(str(train_files[i][1]) + ' ' + str(train_files[j][1]) + ' ')
                diff = random.choice(train_files)
                while train_files[i][0] == diff[0]:
                    diff = random.choice(train_files)
                g.write(str(diff[1]) + '\n')
                j += 1
                if j == len(train_files):
                    break

    test_files = []
    for speaker in test_speakers:
        line = [l for l in lines if l.startswith(speaker)]
        if len(line) > 0:
            tokens = line[0].split()
            test_files.extend(([(speaker, token.split('-')[2]) for token in tokens[1:] if token.split('-')[2].startswith('t')]))

    with open(os.path.join(DATA_DIR, 'test_matches.txt'), 'w') as g:
        for i in range(len(test_files) - 1):
            j = i + 1
            while test_files[i][0] == test_files[j][0]:
                g.write(str(test_files[i][1]) + ' ' + str(test_files[j][1]) + ' ')
                diff = random.choice(test_files)
                while test_files[i][0] == diff[0]:
                    diff = random.choice(test_files)
                g.write(str(diff[1]) + '\n')
                j += 1
                if j == len(test_files):
                    break


def extract_dvector(model, feats):
    vectors = model.predict(feats)
    return np.mean(vectors, axis=0)


def create_metric_train_data(vector_model):
    with open(os.path.join(DATA_DIR, 'train_matches.txt')) as f:
        matches = f.readlines()
    count = 0
    x_train = np.empty((2 * len(matches), 512))
    y_train = np.empty((2 * len(matches), 1))
    for i in range(len(matches)):
        tokens = matches[i].split()
        anch = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[0]))))
        same = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[1]))))
        diff = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[2]))))
        if len(anch) == 0 or len(same) == 0 or len(diff) == 0:
            continue
        x_train[count] = np.concatenate((anch, same))
        y_train[count] = 1
        x_train[count + 1] = np.concatenate((anch, diff))
        y_train[count + 1] = 0
        count += 2
    np.save(os.path.join(METRIC_DATA_DIR, 'x_train'), x_train)
    np.save(os.path.join(METRIC_DATA_DIR, 'y_train'), y_train)


def create_metric_test_data(vector_model):
    with open(os.path.join(DATA_DIR, 'test_matches.txt')) as f:
        matches = f.readlines()
    count = 0
    x_test = np.empty((2 * len(matches), 512))
    y_test = np.empty((2 * len(matches), 1))
    for i in range(len(matches)):
        tokens = matches[i].split()
        anch = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[0]))))
        same = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[1]))))
        diff = extract_dvector(vector_model, np.load(os.path.join(DATA_DIR, 'p1\\{}.npy'.format(tokens[2]))))
        if len(anch) == 0 or len(same) == 0 or len(diff) == 0:
            continue
        x_test[count] = np.concatenate((anch, same)).transpose()
        y_test[count] = 1
        x_test[count + 1] = np.concatenate((anch, diff)).transpose()
        y_test[count + 1] = 0
        count += 2
    np.save(os.path.join(METRIC_DATA_DIR, 'x_test'), x_test)
    np.save(os.path.join(METRIC_DATA_DIR, 'y_test'), y_test)


def train_metric_model():
    model = Sequential()

    model.add(Dense(HIDDEN_SIZE, input_shape=(HIDDEN_SIZE * 2,)))
    model.add(Activation('relu'))

    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.0000001)
    csv_logger = CSVLogger('training_metric.log')

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    x_train = np.load(os.path.join(METRIC_DATA_DIR, 'x_train.npy'))
    y_train = np.load(os.path.join(METRIC_DATA_DIR, 'y_train.npy'))

    model.fit(x_train, y_train, epochs=150, verbose=1, validation_split=0.12, callbacks=[early_stopping, reduce_lr, csv_logger])

    return model


def test_model_dml(metric_model):
    x_test = np.load(os.path.join(METRIC_DATA_DIR, 'x_test.npy'))
    y_test = np.load(os.path.join(METRIC_DATA_DIR, 'y_test.npy'))
    y_pred = metric_model.predict(x_test)
    return test_results(y_test, y_pred)


def test_model_cosine():
    x_test = np.load(os.path.join(METRIC_DATA_DIR, 'x_test.npy'))
    y_test = np.load(os.path.join(METRIC_DATA_DIR, 'y_test.npy'))
    y_pred = [1 - cosine(vectors[:HIDDEN_SIZE], vectors[HIDDEN_SIZE:]) for vectors in x_test]
    return test_results(y_test, y_pred)


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


def test_results(y_true, y_pred):
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


if __name__ == "__main__":
    if not os.path.exists(SPEAKER_DATA_DIR):
        os.mkdir(SPEAKER_DATA_DIR)
    if not os.path.exists(VECTOR_DATA_DIR):
        os.mkdir(VECTOR_DATA_DIR)
    if not os.path.exists(METRIC_DATA_DIR):
        os.mkdir(METRIC_DATA_DIR)
    # print('Extracting features')
    # extract_features(DATA_DIR, NUM_FEATS)
    # print('Creating speaker files')
    # create_speaker_files()
    # print('Creating training and validation data for vector extractor training')
    # create_train_valid_data()
    # print('Training vector extractor model')
    # vector_model = train_vector_model()
    # print('Saving the vector extractor model')
    # vector_model.save('vector_model.h5')
    print('Loading the vector extractor model')
    vector_model = load_model('vector_model.h5')
    print('Matching file couples for metric training the system testing')
    match_couples()
    print('Creating training data for metric training')
    create_metric_train_data(vector_model)
    print('Training the metric model')
    metric_model = train_metric_model()
    print('Saving the metric model')
    metric_model.save('metric_model.h5')
    # print('Loading the metric model')
    # metric_model = load_model('metric_model.h5')
    print('Creating test data for overall system')
    create_metric_test_data(vector_model)
    print('Testing the system')
    eer = test_model_dml(metric_model)
    print('eer: {}'.format(eer))
