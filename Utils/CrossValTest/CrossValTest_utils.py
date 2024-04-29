from sklearn.preprocessing import MinMaxScaler
import warnings
#!pip install natsort
import os
from typing import List
from typing import Tuple
import pandas as pd
import os
import cv2
import glob
import random
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from natsort import natsorted
import optuna
import sys
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(1, parent_dir)

# import functions from hyperParameter Utils
from Utils.HyperparameterSearch.HyperParameter_utils import get_all_csv_paths, batch_X, \
                                                            window, batch_Y, balance_dataset, min_max_norm, \
                                                            horizontal_flip, _horizontal_flip, add_corners, preprocess, \
                                                            batch_all_data, find_components, find_component_intersections, \
                                                            custom_evaluation_hits
from Utils.HyperparameterSearch.GRU import get_GRU

def dataBatcher_test(winlen: int, stepsize: int, num_relax: int,
                     alpha: float, remove_key_points: bool, corners: bool,
                     only_hits: bool = False, only_bounce: bool = False,
                     only_ball: bool = False, only_pose: bool = False):
    data_paths = get_all_csv_paths()

    random.seed(0)
    random.shuffle(data_paths)

    paths_dict = dict()

    test_games = ['game1', 'game8']

    test_paths = []
    _train_paths = []

    game1_paths = []
    game8_paths = []

    for path in data_paths:
        if path.split('/')[-3] in test_games:
            test_paths.append(path)
            if path.split('/')[-3] == 'game1':
                game1_paths.append(path)
            elif path.split('/')[-3] == 'game8':
                game8_paths.append(path)
            # Remove game11
        elif path.split('/')[-3] == 'game11':
            continue
        else:
            _train_paths.append(path)

    # Create train data
    x_train, y_train = batch_all_data(_train_paths, winlen, stepsize, num_relax,
                                      alpha=alpha, remove_key_points=remove_key_points,
                                      corners=corners, only_hits=only_hits, only_bounce=only_bounce,
                                      flip=True, only_ball=only_ball, only_pose=only_pose)

    # Create test data
    x_test, y_test = batch_all_data(test_paths, winlen, stepsize, num_relax,
                                    alpha=alpha, remove_key_points=remove_key_points,
                                    corners=corners, only_hits=only_hits, only_bounce=only_bounce,
                                    flip=False, only_ball=only_ball, only_pose=only_pose)

    # Per game results
    x_game1, y_game1 = batch_all_data(game1_paths, winlen, stepsize, num_relax,
                                      alpha=alpha, remove_key_points=remove_key_points,
                                      corners=corners, only_hits=only_hits, only_bounce=only_bounce,
                                      flip=False, only_ball=only_ball, only_pose=only_pose)

    x_game8, y_game8 = batch_all_data(game8_paths, winlen, stepsize, num_relax,
                                      alpha=alpha, remove_key_points=remove_key_points,
                                      corners=corners, only_hits=only_hits, only_bounce=only_bounce,
                                      flip=False, only_ball=only_ball, only_pose=only_pose)

    # generates a random permutation index
    perm_train = np.random.permutation(len(x_train))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]

    return (x_train, y_train), (x_test, y_test), (x_game1, y_game1), (x_game8, y_game8)


# Utility for running experiments.
def run_experiment_cross_val(filepath, X_train, Y_train, X_val, Y_val, epochs, only_hits,
                             early_stopping_patience=10,
                             n_features=70,
                             num_rnn_layers=2, num_rnn_units=28, n_classes=2, winlen=14, num_relax=6, dropout=0.4,
                             val_paths=[],
                             class_weights: bool = False):
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True,
    #                             save_best_only=True, verbose=-1)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)

    GRU = get_GRU(n_features=n_features,
                  dropout=dropout,
                  num_rnn_layers=num_rnn_layers,
                  num_rnn_units=num_rnn_units,
                  n_classes=n_classes,
                  winlen=winlen,
                  num_relax=num_relax)

    if class_weights:
        _class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
        class_weights_dict = dict(zip(np.unique(Y_train), _class_weights))

        history = GRU.fit(X_train, Y_train,
                          # validation = 0.2,
                          epochs=epochs,
                          class_weight=class_weights_dict,
                          callbacks=[reduce_lr, early_stopping])
    else:
        history = GRU.fit(X_train, Y_train,
                          # validation = 0.2,
                          epochs=epochs,
                          callbacks=[reduce_lr, early_stopping])
    # history = GRU.fit(X_train, Y_train,
    # validation = 0.2,
    #    epochs=epochs,
    #    callbacks=[checkpoint, reduce_lr, early_stopping])

    # GRU.load_weights(filepath)

    probs = GRU.predict(X_val)

    preds = np.argmax(probs, axis=1)

    # Convert indices to actual predictions (0 or 1)
    preds = preds.astype(int)

    # Calculate accuracy, F1, precision, and recall
    accuracy = accuracy_score(Y_val, preds)
    f1 = f1_score(Y_val, preds, average='macro')
    precision_per_class = precision_score(Y_val, preds, average=None)
    recall_per_class = recall_score(Y_val, preds, average=None)

    # custom eval metrics
    preds_hits_only = np.where(preds == 2, 0, preds)
    true_hits_only = np.where(Y_val == 2, 0, Y_val)

    preds_bounces_only = np.where(preds == 1, 0, preds)
    preds_bounces_only = np.where(preds_bounces_only == 2, 1, preds_bounces_only)
    true_bounces_only = np.where(Y_val == 1, 0, Y_val)
    true_bounces_only = np.where(true_bounces_only == 2, 1, true_bounces_only)

    if only_hits:
        missing_hits, wrong_hits, overlap_hits = custom_evaluation_hits(true_hits_only, preds_hits_only)
        missing_bounces, wrong_bounces, overlap_bounces = 0, 0, 0
    else:
        missing_hits, wrong_hits, overlap_hits = custom_evaluation_hits(true_hits_only, preds_hits_only)
        missing_bounces, wrong_bounces, overlap_bounces = custom_evaluation_hits(true_bounces_only, preds_bounces_only)

    conf_matrix = confusion_matrix(Y_val, preds)

    return_dict = {'accuracy': accuracy,
                   'f1': f1,
                   'precision': precision_per_class.tolist(),
                   'recall': recall_per_class.tolist(),
                   'missing_hits': missing_hits,
                   'wrong_hits': wrong_hits,
                   'overlap_hits': overlap_hits,
                   'missing_bounces': missing_bounces,
                   'wrong_bounces': wrong_bounces,
                   'overlap_bounces': overlap_bounces,
                   'confusion_matrix': conf_matrix.tolist(),
                   'preds': preds.tolist(),
                   'Y_val': Y_val.tolist(),
                   'history': history.history['loss'],
                   'validation_paths': val_paths}

    # prec, recall, acc, f1 = eval_metrics(Y_val, preds)

    # true_labels = Y_test

    # accuracy = accuracy_score(preds, Y_test)

    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test f1: {round(f1 * 100, 2)}%")
    # print(f"Test precision: {precision_per_class}")
    # print(f"Test recall: {recall_per_class}")
    # print(f"Test missing hits: {round(missing_hits * 100, 2)}%")
    # print(f"Test wrong hits: {round(wrong_hits * 100, 2)}%")
    # print(f"Test overlap hits: {round(overlap_hits * 100, 2)}%")
    # print(f"Test missing bounces: {round(missing_bounces * 100, 2)}%")
    # print(f"Test wrong bounces: {round(wrong_bounces * 100, 2)}%")
    # print(f"Test overlap bounces: {round(overlap_bounces * 100, 2)}%")

    return return_dict, GRU


def cross_val(winlen: int, stepsize: int, num_relax: int,
              alpha: float, remove_key_points: bool,
              corners: bool, only_hits: bool, dropout: float,
              num_rnn_layers: int, num_rnn_units: int,
              n_splits: int, only_ball: bool, only_pose: bool,
              epochs: int, class_weights: bool):
    paths = get_all_csv_paths()
    games = ['game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game9', 'game10']
    k_fold = KFold(n_splits=n_splits)

    return_results = dict()
    return_dicts = []
    histories = []

    for train_indices, val_indices in k_fold.split(games):

        train_games = [games[i] for i in train_indices]
        val_games = [games[i] for i in val_indices]

        train_paths = []
        val_paths = []

        for path in paths:

            if path.split('/')[-3] in train_games:
                train_paths.append(path)
            elif path.split('/')[-3] in val_games:
                val_paths.append(path)

        X_train, Y_train = batch_all_data(train_paths, winlen=winlen, stepsize=stepsize,
                                          num_relax=num_relax, alpha=alpha,
                                          remove_key_points=remove_key_points,
                                          corners=corners, only_hits=only_hits,
                                          flip=True, only_ball=only_ball,
                                          only_pose=only_pose)

        X_val, Y_val = batch_all_data(val_paths, winlen=winlen, stepsize=stepsize,
                                      num_relax=num_relax, alpha=alpha,
                                      remove_key_points=remove_key_points,
                                      corners=corners, only_hits=only_hits,
                                      flip=False, only_ball=only_ball,
                                      only_pose=only_pose)

        weights_path = 'hits_GRU.weights.h5'
        n_features = 70

        if remove_key_points:
            n_features -= 20
        if corners:
            n_features += 8

        if only_ball:
            n_features -= 68

        if only_pose:
            n_features -= 2

        if only_hits:
            n_classes = 2
        else:
            n_classes = 3

        return_metrics, GRU = run_experiment_cross_val(filepath=weights_path,
                                                       X_train=X_train, Y_train=Y_train,
                                                       X_val=X_val, Y_val=Y_val,
                                                       epochs=epochs,
                                                       n_features=n_features,
                                                       dropout=dropout,
                                                       num_rnn_layers=num_rnn_layers,
                                                       num_rnn_units=num_rnn_units,
                                                       n_classes=n_classes,
                                                       winlen=winlen,
                                                       num_relax=num_relax,
                                                       val_paths=val_paths,
                                                       class_weights=class_weights,
                                                       only_hits=only_hits)

        return_dicts.append(return_metrics)

        for key, value in return_metrics.items():

            if key in return_results:
                return_results[key].append(value)
            else:
                return_results[key] = [value]

        # histories.append(history)

    return return_results, return_dicts, GRU  # , histories


def mean_results(results):
    mean_dict = dict()

    for key, value in results.items():

        if key == 'preds' or key == 'Y_val' or key == 'validation_paths':
            continue

        # Calculate average
        mean_dict[key + '_mean'] = np.mean(results[key], axis=0).tolist()

        # Calculate standard deviation
        mean_dict[key + '_std'] = np.std(results[key], axis=0).tolist()

    return mean_dict




def run_and_save(folder_name: str, winlen: int, stepsize: int, num_relax: int,
                 alpha: float, remove_key_points: bool, corners: bool, only_hits: bool, dropout: float,
                 num_rnn_layers: int,
                 num_rnn_units: int, only_ball: bool, only_pose: bool, val_epochs: int, test_epochs,
                 class_weights: bool):
    # Cross Validation Results
    results, return_dicts, _ = cross_val(winlen=winlen, stepsize=stepsize, num_relax=num_relax,
                                         alpha=alpha, remove_key_points=remove_key_points,
                                         corners=corners, only_hits=only_hits, dropout=dropout,
                                         num_rnn_layers=num_rnn_layers, num_rnn_units=num_rnn_units,
                                         n_splits=4, only_ball=only_ball, only_pose=only_pose,
                                         epochs=val_epochs, class_weights=class_weights)

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # Create a df for each split
    for i in range(len(return_dicts)):
        dict_path = os.path.join(folder_name, 'split_' + str(i) + '_val.json')

        with open(dict_path, 'w') as json_file:
            json.dump(return_dicts[i], json_file)

    final_val_results = mean_results(results)

    cross_val_results_path = os.path.join(folder_name, 'cross_val_result.json')

    with open(cross_val_results_path, 'w') as json_file:
        json.dump(final_val_results, json_file)

    # Test Results

    (final_X_train, final_Y_train), (final_X_test, final_Y_test), (x_game1, y_game1), (
    x_game8, y_game8) = dataBatcher_test(winlen=winlen,
                                         stepsize=stepsize,
                                         num_relax=num_relax,
                                         alpha=alpha,
                                         remove_key_points=remove_key_points,
                                         corners=corners,
                                         only_pose=only_pose,
                                         only_ball=only_ball,
                                         only_hits=only_hits,
                                         only_bounce=False)

    weights_path = 'Gru_test.weights.h5'
    n_features = 70

    if remove_key_points:
        n_features -= 20
    if corners:
        n_features += 8

    if only_ball:
        n_features -= 68

    if only_pose:
        n_features -= 2

    if only_hits:
        n_classes = 2
    else:
        n_classes = 3

    return_metrics, GRU = run_experiment_cross_val(filepath=weights_path,
                                                   X_train=final_X_train, Y_train=final_Y_train,
                                                   X_val=final_X_test, Y_val=final_Y_test,
                                                   epochs=test_epochs,
                                                   n_features=n_features,
                                                   dropout=dropout,
                                                   num_rnn_layers=num_rnn_layers,
                                                   num_rnn_units=num_rnn_units,
                                                   n_classes=n_classes,
                                                   winlen=winlen,
                                                   num_relax=num_relax,
                                                   class_weights=class_weights,
                                                   only_hits=only_hits)

    # Make test results on game basis

    # game1
    for i, (x_game, y_game) in enumerate([[x_game1, y_game1], [x_game8, y_game8]]):
        probs = GRU.predict(x_game)
        preds = np.argmax(probs, axis=1)
        preds = preds.astype(int)
        accuracy = accuracy_score(y_game, preds)
        f1 = f1_score(y_game, preds, average='macro')
        # custom eval metrics
        preds_hits_only = np.where(preds == 2, 0, preds)
        true_hits_only = np.where(y_game == 2, 0, y_game)

        preds_bounces_only = np.where(preds == 1, 0, preds)
        preds_bounces_only = np.where(preds_bounces_only == 2, 1, preds_bounces_only)
        true_bounces_only = np.where(y_game == 1, 0, y_game)
        true_bounces_only = np.where(true_bounces_only == 2, 1, true_bounces_only)

        if only_hits:
            missing_hits, wrong_hits, overlap_hits = custom_evaluation_hits(true_hits_only, preds_hits_only)
            missing_bounces, wrong_bounces, overlap_bounces = 0, 0, 0
        else:
            missing_hits, wrong_hits, overlap_hits = custom_evaluation_hits(true_hits_only, preds_hits_only)
            missing_bounces, wrong_bounces, overlap_bounces = custom_evaluation_hits(true_bounces_only, preds_bounces_only)

        game_dict = {'accuracy': accuracy,
                     'f1': f1,
                     'missing_hits': missing_hits,
                     'wrong_hits': wrong_hits,
                     'overlap_hits': overlap_hits,
                     'missing_bounces': missing_bounces,
                     'wrong_bounces': wrong_bounces,
                     'overlap_bounces': overlap_bounces}

        test_game_results_path = os.path.join(folder_name, f'game{i * 7 + 1}.json')
        with open(test_game_results_path, 'w') as json_file:
            json.dump(game_dict, json_file)

    test_results_path = os.path.join(folder_name, 'test_result.json')

    with open(test_results_path, 'w') as json_file:
        json.dump(return_metrics, json_file)

    test_weights_path = os.path.join(folder_name, 'test.weights.h5')

    GRU.save_weights(test_weights_path)

    return None




