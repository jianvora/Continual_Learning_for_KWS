from os import listdir
from os.path import join, isfile, isdir, normpath
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from itertools import zip_longest
import librosa
import random
import math
import numpy as np
from random import seed, randint
from train import get_tc_resnet_8, get_tc_resnet_14
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, ReLU, BatchNormalization, Add, AveragePooling1D, Dense, Flatten, Dropout, Activation
from google.colab import files

# To train entire tc-resnet--uncomment
"""
from process_data import process_file, generate_noisy_sample
def __load_audio_filenames_with_class__(root_folder):
    classes = [item for item in listdir(root_folder) if isdir(
        join(root_folder, item)) and not item.startswith('_')]
    filenames = []
    class_ids = []
    for i in range(len(classes)):
        c = classes[i]
        class_filenames = __load_audio_filenames__(join(root_folder, c))
        filenames.extend(class_filenames)
        class_ids.extend([i] * len(class_filenames))
    return filenames, class_ids, classes


def __load_audio_filenames__(root_folder):
    filenames = []
    for entry in listdir(root_folder):
        full_path = join(root_folder, entry)
        if (isfile(full_path)):
            if (entry.endswith('.wav')):
                filenames.append(full_path)
        else:
            filenames.extend(__load_audio_filenames__(full_path))
        if (len(filenames) >= 100):
            break
    return filenames


def __load_subset_filenames__(root_folder, filename):
    subset_list = []
    with open(join(root_folder, filename)) as f:
        for line in f:
            line = line.strip()
            if (len(line) == 0):
                continue
            subset_list.append(normpath(join(root_folder, line)))
    return set(subset_list)


def load_data_from_folder(root_folder):
    filenames, class_ids, classes = __load_audio_filenames_with_class__(root_folder)
    dataset_size = len(filenames)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_validation = []
    y_validation = []
    pool = Pool(cpu_count() - 1)
    for (results, filepath, class_id, random_roll) in tqdm(pool.imap_unordered(process_file, zip_longest(filenames, class_ids)), total=dataset_size):
        filepath = normpath(filepath)
        is_testing = 1 <= random_roll and random_roll <= 10
        is_validation = 11 <= random_roll and random_roll <= 20
        for item in results:
            if (is_testing):
                X_test.append(item)
                y_test.append(class_id)
            elif (is_validation):
                X_validation.append(item)
                y_validation.append(class_id)
            else:
                X_train.append(item)
                y_train.append(class_id)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)
    return X_train, y_train, X_test, y_test, X_validation, y_validation, classes

X_train, y_train, X_test, y_test, X_validation, y_validation, classes = load_data_from_folder('dataset')
num_classes = len(classes)
print(num_classes)
(num_train, input_length, num_channel) = X_train.shape
num_test = X_test.shape[0]
num_validation = X_validation.shape[0]
model_14 = get_tc_resnet_8((input_length, num_channel), num_classes, 1.5)
model_14.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#checkpoint_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', save_weights_only=True, period=5)
model_14.fit(x=X_train, y=y_train, batch_size=1024, epochs=100, validation_data=(X_test, y_test))#, callbacks=[checkpoint_cb])
print(model_14.evaluate(X_validation, y_validation))
model_14.save_weights('weights.h5')
files.download('weights.h5') 
"""


AUDIO_LENGTH = 2

seed(163)


def __load_background_noises__(root_folder):
    noises = []
    noise_folder = join(root_folder, '_background_noise_')
    for item in listdir(noise_folder):
        if (not item.endswith('.wav')):
            continue
        samples, sr = librosa.load(join(noise_folder, item), sr=None)
        noises.append(samples)
    return noises

def generate_noisy_sample(samples, noise):
    samples_length = len(samples)
    noise_length = len(noise)
    if (noise_length < samples_length):
        return samples
    noise_start = random.randint(0, noise_length - samples_length - 1)
    noise_part = noise[noise_start:noise_start + samples_length]
    noise_coeff = random.uniform(0.0, 0.1)
    audio_offset = math.floor(
        random.uniform(-samples_length * 0.1, samples_length * 0.1))
    new_samples = np.zeros((samples_length))
    if (audio_offset >= 0):
        new_samples[audio_offset:] = samples[:samples_length - audio_offset]
    else:
        new_samples[:samples_length + audio_offset] = samples[-audio_offset:]
    new_samples = noise_part * noise_coeff +         (1.0 - noise_coeff) * new_samples
    return new_samples


def get_mfcc(samples, sr):
    return librosa.feature.mfcc(samples, sr=sr, n_mfcc=40, n_fft=400, hop_length=100).transpose()


def process_file(argv):
    (filepath, class_id) = argv
    results = []
    samples, sr = librosa.load(filepath, sr=None)
    samples_len = len(samples)
    if (samples_len > sr * AUDIO_LENGTH):
        samples = samples[- sr * AUDIO_LENGTH:]
    elif (samples_len < sr * AUDIO_LENGTH):
        temp = np.zeros((sr * AUDIO_LENGTH))
        temp[:samples_len] = samples
        samples = temp
    mfcc = get_mfcc(samples, sr)
    results.append(mfcc)
    random_roll = randint(1, 100)
    is_testing = 1 <= random_roll and random_roll <= 10
    is_validation = 11 <= random_roll and random_roll <= 20
    if (not is_testing and not is_validation):
        for item in noises:
            new_samples = generate_noisy_sample(samples, item)
            mfcc = get_mfcc(new_samples, sr)
            results.append(mfcc)
    return results, filepath, class_id, random_roll

def test_process_file(argv):
    (filepath) = argv
    results = []
    samples, sr = librosa.load(filepath, sr=None)
    samples_len = len(samples)
    if (samples_len > sr * AUDIO_LENGTH):
        samples = samples[- sr * AUDIO_LENGTH:]
    elif (samples_len < sr * AUDIO_LENGTH):
        temp = np.zeros((sr * AUDIO_LENGTH))
        temp[:samples_len] = samples
        samples = temp
    mfcc = get_mfcc(samples, sr)
    results.append(mfcc)
    return results, filepath


def __load_new_audio_filenames_with_class__(root_folder):
    classes = [item for item in listdir(root_folder) if item.startswith('T')] #classes as "T00xx"
    filenames = []
    class_ids = []
    for i in range(len(classes)):
        c = classes[i] 
        class_filenames = __load_new_audio_filenames__((join(root_folder, c, "enrollment"))) #location of wav files for kws
        filenames.extend(class_filenames)
        class_ids.extend([i] * len(class_filenames))
    return filenames, class_ids, classes


def __load_new_audio_filenames__(root_folder):
    filenames = []
    for entry in listdir(root_folder):
        full_path = join(root_folder, entry)
        if (isfile(full_path)):
            if (entry.endswith('.wav')):
                filenames.append(full_path)
        else:
            filenames.extend(__load_audio_filenames__(full_path))
        #if (len(filenames) >= 10):
            #break
    return filenames

def load_new_data_from_folder(root_folder):
    filenames, class_ids, classes = __load_new_audio_filenames_with_class__(root_folder)
    dataset_size = len(filenames)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_validation = []
    y_validation = []
    pool = Pool(cpu_count() - 1)
    for (results, filepath, class_id, random_roll) in tqdm(pool.imap_unordered(process_file, zip_longest(filenames, class_ids)), total=dataset_size):
        filepath = normpath(filepath)
        is_testing = 1 <= random_roll and random_roll <= 10 #vary these to modify train-val-test lengths
        is_validation = 11 <= random_roll and random_roll <=20
        for item in results:
            if (is_testing):
                X_test.append(item)
                y_test.append(class_id)
            elif (is_validation):
                X_validation.append(item)
                y_validation.append(class_id)
                X_test.append(item)
                y_test.append(class_id)
            else:
                X_train.append(item)
                y_train.append(class_id)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)
    return X_train, y_train, X_test, y_test, X_validation, y_validation, classes

def final_func():

    original_model    = get_tc_resnet_8((321, 40), 30, 1.5) #model corresponding to kws on google speech cmds: input length, num_channel, num_classes
    original_model.load_weights('weights.h5') #Assuming this file is loaded in the current working dir
    bottleneck_input  = original_model.get_layer(index=0).input
    print(bottleneck_input)
    bottleneck_output = original_model.get_layer(index=-2).output
    print(bottleneck_output)
    bottleneck_model  = Model(inputs=bottleneck_input,outputs=bottleneck_output)



    # Add the last softmax layer
    for layer in bottleneck_model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(bottleneck_model)
    kws_classes = 100
    new_model.add(Dense(kws_classes, activation="softmax", input_dim=2808))

    noises = __load_background_noises__('dataset1/train')

    X_train, y_train, X_test, y_test, X_validation, y_validation, classes = load_new_data_from_folder('dataset1/train')
    num_classes = len(classes)
    (num_train, input_length, num_channel) = X_train.shape
    num_test = X_test.shape[0]
    num_validation = X_validation.shape[0]
    print(num_classes)
    print(num_train)
    print(num_test)
    print(num_validation)

    new_model.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(x=X_train, y=y_train, batch_size=512, epochs=500, validation_data=(X_test, y_test))
    print(new_model.evaluate(X_validation, y_validation))
    new_model.save_weights('new_weights.h5')

    y_pred = new_model.predict(X_test)
    
    return y_pred, y_test

def kws_final_func(test_dir):
    original_model    = get_tc_resnet_8((321, 40), 30, 1.5) #model corresponding to kws on google speech cmds: input length, num_channel, num_classes
    bottleneck_input  = original_model.get_layer(index=0).input
    bottleneck_output = original_model.get_layer(index=-2).output
    bottleneck_model  = Model(inputs=bottleneck_input,outputs=bottleneck_output)
    for layer in bottleneck_model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(bottleneck_model)
    kws_classes = 100
    new_model.add(Dense(kws_classes, activation="softmax", input_dim=2808))
    new_model.load_weights('new_weights.h5') #pre-trained model

    # Test data
    X_test = []
    filenames = []
    class_filenames = __load_new_audio_filenames__(test_dir) #location of wav files for kws
    filenames.extend(class_filenames)
    dataset_size = len(filenames)
    for (results, filepath) in tqdm(pool.imap_unordered(test_process_file, zip_longest(filenames)), total=dataset_size):
        for item in results:
            X_test.append(item)
    X_test = np.array(X_test)
    
    y_pred = new_model.predict(X_test)

    return y_pred




    



