import cv2
import pandas as pd
import imageio
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import glob
import os
from tensorflow.keras.models import load_model, save_model

from Utils.HyperparameterSearch.GRU import get_GRU

from Utils.HyperparameterSearch.HyperParameter_utils import batch_X, window, batch_Y, min_max_norm, preprocess

def batch_duel(df: pd.DataFrame, winlen: int, stepsize: int,
               num_relax: int, remove_key_points: bool, corners: bool):
    num_features = len(df.columns) - 2

    nr_features_to_remove = 0

    if remove_key_points:
        nr_features_to_remove += 20
    if corners:
        nr_features_to_remove -= 8

    all_x = np.array([]).reshape((0, winlen, num_features - nr_features_to_remove))
    all_y = np.array([]).reshape((0,))

    (x, y), _ = preprocess(df, flip=False, alpha=0.5, remove_key_points=remove_key_points)

    # batch the data
    batch_x = batch_X(x, winlen, stepsize)
    batch_y = batch_Y(y, winlen, stepsize, num_relax)

    all_x = np.concatenate((all_x, batch_x))
    all_y = np.concatenate((all_y, batch_y))
    return (all_x, all_y)


def modify_Y_inference(Y_inference: np.array, winlen, num_relax) -> np.array:
    # winlen - relax
    # relax - 1

    equalized_Y = np.append(np.array([0 for i in range(winlen - num_relax)]), Y_inference)
    equalized_Y = np.append(equalized_Y, np.array([0 for i in range(num_relax - 1)]))

    return equalized_Y


def load_GRU(model_path):
    #     model_path = r"C:\Users\sofu0\PycharmProjects\BadmintonTDK-SoGuMo\models\GRU\best\model"

    n_features = 50
    num_rnn_layers = 8
    num_rnn_units = 32
    dropout = 0.1
    n_classes = 3
    winlen = 21

    GRU = get_GRU(num_rnn_layers=num_rnn_layers, num_rnn_units=num_rnn_units, dropout=dropout, winlen=winlen,
                  n_classes=n_classes, n_features=n_features)
    GRU.load_weights(model_path)

    return GRU


def return_predictions(data_path, model_path, winlen, num_relax, remove_key_points=True):
    # GRU = load_GRU(model_path)
    GRU = load_model(model_path)

    data = pd.read_csv(data_path)
    X_inference, _ = batch_duel(df=data, winlen=winlen, stepsize=1,
                                num_relax=num_relax, remove_key_points=remove_key_points, corners=False)

    probabilities = GRU.predict(X_inference)
    pred_label = np.argmax(probabilities, axis=1)

    #     return pred_label
    Y_preds = modify_Y_inference(pred_label, winlen=winlen, num_relax=num_relax)

    #     Y_preds = custom_optimizaiton(Y_preds)
    return Y_preds


def get_hits(duel: np.array):
    i = 0
    hits = {"hit": {"start": [], "end": []}, "bounce": {"start": [], "end": []}}
    while i < len(duel):

        previous_element = duel[i - 1]
        element = duel[i]

        if element == 1 and previous_element != 1:
            hits["hit"]["start"].append(i)

        if element != 1 and previous_element == 1:
            hits["hit"]["end"].append(i)

        if element == 2 and previous_element != 2:
            hits["bounce"]["start"].append(i)

        if element != 2 and previous_element == 2:
            hits["bounce"]["end"].append(i)

        i += 1

    return hits


from typing import Dict


def get_shot(hits: Dict):
    """

    Converts the dictionary of hits, into a dictionary of shots. The key will be shot id, and the values will be
    start and end frame, where the endframe will be the frame before the next start
    """

    shots = {}
    start_times = hits["hit"]["start"]

    for i in range(len(start_times) - 1):
        shots[i] = {}
        shots[i]["start"] = start_times[i]
        shots[i]["end"] = start_times[i + 1] - 1

    return shots


def idenity_bounces_between_shots(hits: Dict):
    shot_to_bounce = {}

    for i in range(len(hits['hit']['end'])):

        bounce_found = False

        shot_to_bounce[i] = {}
        shot_to_bounce[i]["start"] = hits['hit']['end'][i]

        if i != len(hits['hit']['end']) - 1:

            frames_in_hit = [i for i in range(hits['hit']['end'][i], hits['hit']['end'][i + 1])]

            for j in hits['bounce']['end']:

                if j in frames_in_hit:
                    bounce_found = True
                    shot_to_bounce[i]["end"] = j - 1
                    break

            if not bounce_found:
                shot_to_bounce[i]["end"] = hits['hit']['end'][i + 1]
        else:

            for j in hits['bounce']['end']:

                if j > hits['hit']['end'][i]:
                    bounce_found = True
                    shot_to_bounce[i]["end"] = j - 1
                    continue

            if not bounce_found:
                del shot_to_bounce[i]

    return shot_to_bounce


def extract_shot_WASB(game: str, clip: str, start_frame: int, end_frame: int):
    ball_path = f"/kaggle/input/filtered-poses/Dataset/{game}/{clip}/data.csv"
    true_path = f"/kaggle/input/tracknet-tennis/Dataset/{game}/{clip}/Label.csv"

    df_WASB = pd.read_csv(ball_path)
    df_WASB = df_WASB.iloc[start_frame: end_frame]

    df_true = pd.read_csv(true_path)
    df_true = df_true.iloc[start_frame: end_frame]

    return (df_WASB['ball_x'].tolist(), df_WASB['ball_y'].tolist()), (
    df_true['x-coordinate'].tolist(), df_true['y-coordinate'].tolist())

def cool_plotter(preds, y_true, modelName):

    acc =  accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='weighted')
    yss = [preds, y_true]
    tickslabels = ["predicted", "true"]

    alphas = [0.5, 1]
    plt.figure(figsize=(50, 7))

    # Set the font size for the title, labels, and ticks
    plt.rcParams.update({'font.size': 32})

    for j in range(2):
        q = get_hits(yss[j])

        for i in range(len(q["hit"]["start"])):
            start = q["hit"]["start"][i]
            end = q["hit"]["end"][i]
            width = end-start
            if j == 1 and i == len(q["hit"]["start"])-1:
                plt.barh(["hit " + tickslabels[j]], width=width, height=0.3, left=start, color='red', alpha=alphas[j], label="player near")
            else:
                plt.barh(["hit " + tickslabels[j]], width=width, height=0.3, left=start, color='red', alpha=alphas[j])

    for j in range(2):
        q = get_hits(yss[j])
        for i in range(len(q["bounce"]["start"])):
            start = q["bounce"]["start"][i]
            end = q["bounce"]["end"][i]
            width = end-start
            if j == 1 and i == len(q["bounce"]["start"])-1:
                plt.barh(["bounce " + tickslabels[j]], width=width, height=0.3, left=start, color='blue', alpha=alphas[j], label="player far")
            else:
                plt.barh(["bounce " + tickslabels[j]], width=width, height=0.3, left=start, color='blue', alpha=alphas[j])

    # Create a list of x-coordinates for the bars
    x = list(range(len(preds)))
    # Create a list of y-coordinates for the bars
    # y = [i if pred_label[i] in [1,2] else None for i in x]

    # Set the limits of the x-axis and y-axis
    plt.xlim(-0.5, len(preds)+100)
    # plt.ylim(0.5, 3.5)
    # Remove the y-axis ticks and labels
    plt.yticks(["hit predicted", "hit true", "bounce predicted", "bounce true"])
    # plt.ylabel(' ')
    plt.title(modelName + " - Acc :" +  str(round(acc, 2)) + " and F1: " + str(round(f1, 2)))
    plt.legend()
    plt.savefig("duel21.png")

    plt.show()


def show_shot_on_video(df, shot_id, video_path):
    game = df.iloc[shot_id]['game']
    clip = df.iloc[shot_id]['clip']

    start = df.iloc[shot_id]['start']
    end = df.iloc[shot_id]['end']

    print(clip, game)

    images_path = f'/kaggle/input/tracknet-tennis/Dataset/{game}/{clip}/'

    images = glob.glob(images_path + '/*.jpg')
    images.sort()
    output_video_path = video_path
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))

    frames_for_video = images[start: end]

    for i, frame in enumerate(frames_for_video):
        img = cv2.imread(frame)

        x = df.iloc[shot_id]['x_WASB'][i]
        y = df.iloc[shot_id]['y_WASB'][i]

        cv2.circle(img, (x, y), 10, [255, 0, 0], -1)

        output_video.write(img)

    output_video.release()