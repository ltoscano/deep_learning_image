'''
Given an input folder with annotated images (images are annotated by the folder
to which they belong too), it trains a deep learning algorithm that is able
to classify the images.

Created on Sep 27, 2016

@author: dearj019
'''
import sys
import json
import os
import random
import numpy as np
from keras_deep_learning.models.models_factory import model_factory

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K

import h5py

def generator_from_file_list(image_list, labels, batch_size):
    labels.sort()
    labels_map = {}
    for i in range(len(labels)):
        label = labels[i]
        labels_map[label] = i
        
    while 1:
        random.shuffle(image_list)
        X = []
        Y = []
        for i in range(len(image_list)):
            entry = image_list[i]
            image_path = entry[0]
            label = entry[1]
            
            x = np.load(image_path)
            if x.shape != (1, 3, 224, 224):
                continue
            x = np.squeeze(x)
            
            y = np.zeros((len(labels,)))
            y[labels_map[label]] = 1
            
            if len(X) == batch_size:
                yield np.array(X), np.array(Y)
                X = []
                Y = []
            
            X.append(x)
            Y.append(y)

def set_weights(model, weights_path):
    assert os.path.exists(weights_path)

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers) - 1:
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    
    return model

def do_fold(model, weights_path, lr, regr, fold, checkpoints_folder, training, labels, batch_size,
            nb_epoch, validation):
    set_weights(model, weights_path)
    K.set_value(model.optimizer.lr, lr)
    checkpoint_name = ['/weights', str(fold), str(lr), str(regr)]
    checkpoint_name = "_".join(checkpoint_name)
    checkpoint_name = checkpoints_folder + checkpoint_name + ".hdf5"
    checkpointer = ModelCheckpoint(filepath = checkpoint_name, verbose = 1, save_best_only = True)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0, mode = 'auto')
    
    model.fit_generator(generator = generator_from_file_list(training, labels, batch_size),
                        samples_per_epoch = len(training),
                        nb_epoch = nb_epoch,
                        validation_data = generator_from_file_list(validation, labels, batch_size),
                        show_accuracy = True,
                        verbose = 1,
                        nb_val_samples = len(validation),
                        max_q_size = 200,
                        callbacks = [checkpointer, early_stopping])

def create_training_testing_lists(input_folder, training_validation_split):
    training = []
    validation = []
    labels = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    labels.sort()
    
    for i in range(len(labels)):
        label = labels[i]
        images = [str(a) for a in os.listdir(os.path.join(input_folder, label)) if os.path.isfile(os.path.join(input_folder, label, a))]
        random.shuffle(images)
        cutoff = int(training_validation_split * len(images))
        
        for image_name in images[:cutoff]:
            validation.append((os.path.join(input_folder, label, image_name), label))
        
        for image_name in images[cutoff:]:
            training.append((os.path.join(input_folder, label, image_name), label))
    
    return (training, validation)
    
def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    input_folder = conf['input_folder']
    nb_folds = conf['nb_folds']
    weights_path = conf['weights_path']
    checkpoints_folder = conf['checkpoints_folder']
    lr = conf['lr']
    regr = conf['regr']
    batch_size = conf['batch_size']
    nb_epoch = conf['nb_epoch']
    training_validation_split = conf['training_validation_split']
    model_type = conf['model_type']
    
    labels = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    #1. Create training and validation lists
    training_list, validation_list = create_training_testing_lists(input_folder, training_validation_split)

    model = model_factory(model_type, lr, regr, len(labels))
    
    #2. Create model
    do_fold(model, weights_path, lr, regr, 0, checkpoints_folder, training_list, labels, batch_size,
            nb_epoch, validation_list)

if __name__ == "__main__":
    main()