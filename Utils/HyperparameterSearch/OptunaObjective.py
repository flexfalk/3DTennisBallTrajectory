from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import os
from HyperParameter_utils import *
from GRU import get_GRU
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def objective(trial, n_classes):
    # Parameters for Data
    use_flip = trial.suggest_categorical('use_flip', [True, False])
    alpha = trial.suggest_float('alpha', 0.2, 0.7, step=0.1)

    use_balance = trial.suggest_categorical('use_balance', [True, False])

    #     remove_keypoints = trial.suggest_categorical('remove_keypoints', [True, False])
    #     use_corners = trial.suggest_categorical('use_corners', [True, False])

    winlen_relax_choices = [[14, 6], [7, 3], [21, 9]]
    winlen_relax = trial.suggest_categorical('winlen_relax', winlen_relax_choices)

    # Parse the string back to a tuple in the objective function
    # winlen_relax = tuple(map(int, winlen_relax_str.split(',')))

    winlen, num_relax = winlen_relax
    #     print(winlen, num_relax)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), path_dir = dataBatcher(winlen=winlen, n_split=0.80,
                                                                                 stepsize=1,
                                                                                 num_relax=num_relax,
                                                                                 balance=use_balance, alpha=alpha,
                                                                                 remove_key_points=False, corners=False,
                                                                                 only_hits=False, flip=use_flip)

    # Parameters for model
    use_class_weighs = trial.suggest_categorical('use_class_weights', [True, False])

    rnn_layers = trial.suggest_int('rnn_layers', 2, 8, step=2)
    rnn_units = trial.suggest_int('rnn_units', 16, 32, step=4)
    dropout = trial.suggest_float('dropout', 0.1, 0.6, step=0.1)

    n_features = 70

    #     if remove_keypoints:
    #         n_features -= 20
    #     if use_corners:
    #         n_features += 8

    GRU = get_GRU(n_features=n_features,
                  dropout=dropout,
                  num_rnn_layers=rnn_layers,
                  num_rnn_units=rnn_units,
                  n_classes=n_classes,
                  winlen=winlen,
                  num_relax=num_relax)

    if use_class_weighs:
        _class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
        class_weights_dict = dict(zip(np.unique(Y_train), _class_weights))

    else:
        class_weights_dict = False

    # do something with filepath of checkpoint
    trial_dir = f'trial_{trial.number}'
    checkpoint_path = os.path.join(trial_dir, 'weights.h5')

    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                 save_best_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = GRU.fit(X_train, Y_train,
                      validation_data=(X_val, Y_val),
                      epochs=100,
                      class_weight=class_weights_dict,
                      callbacks=[checkpoint, reduce_lr, early_stopping],
                      verbose=0)

    val_loss = min(history.history['val_loss'])

    # Make predictions on the validation set
    Y_pred = GRU.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Y_val_classes = np.argmax(Y_val, axis=1)

    # Calculate accuracy and F1-score
    accuracy = accuracy_score(Y_val, Y_pred_classes)
    f1_we = f1_score(Y_val, Y_pred_classes, average='weighted')
    f1_mi = f1_score(Y_val, Y_pred_classes, average='micro')
    f1_ma = f1_score(Y_val, Y_pred_classes, average='macro')

    # Calculate precision and recal
    precision_per_class = precision_score(Y_val, Y_pred_classes, average=None)
    recall_per_class = recall_score(Y_val, Y_pred_classes, average=None)

    # custom eval metrics

    missing_hits, wrong_hits, overlap = custom_evaluation_hits(Y_val, Y_pred_classes)

    # Store metrics in user attrs
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('f1_we', f1_we)
    trial.set_user_attr('f1_mi', f1_mi)
    trial.set_user_attr('f1_ma', f1_ma)
    trial.set_user_attr('precision', precision_per_class)
    trial.set_user_attr('recall', recall_per_class)

    trial.set_user_attr('missing_hits', missing_hits)
    trial.set_user_attr('wrong_hits', wrong_hits)
    trial.set_user_attr('overlap', overlap)

    return val_loss