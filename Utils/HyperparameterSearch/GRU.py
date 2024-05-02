import keras
from sklearn.metrics import accuracy_score


def get_GRU(num_rnn_layers, num_rnn_units,
            dropout, winlen, n_classes, n_features):
    features_input = keras.Input((winlen, n_features))

    x = keras.layers.Dense(32, activation="relu")(features_input)

    for i in range(num_rnn_layers - 1):
        x = keras.layers.GRU(num_rnn_units, return_sequences=True)(x)
        x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.GRU(num_rnn_units)(x)

    x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(n_classes, activation="softmax")(x)

    rnn_model = keras.Model(features_input, output)

    rnn_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    return rnn_model