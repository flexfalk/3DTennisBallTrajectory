from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from typing import List
from typing import Tuple
import pandas as pd
import os
import cv2
import glob
import random
import numpy as np


def get_all_csv_paths(root_path: str = "/kaggle/input/filtered-poses/Dataset", corners: bool = False) -> List[str]:
    """
    Gets a list of paths for every data csv

    root_path: string of your root path to the data. For sofus that is
    returns: list
    """

    all_csv = []
    games = glob.glob(root_path + "/*")

    for game in games:
        clips = glob.glob(game + "/*")
        for clip in clips:
            all_csv.append(clip + "/data.csv")

    return all_csv


def batch_X(df: pd.DataFrame, winlen: int, stepsize: int):
    array = df.to_numpy()
    # window
    window_X = window(array, winlen, stepsize)

    return window_X


def window(array: np.array, winlen: int, stepsize: int):
    sub_windows = (
            np.expand_dims(np.arange(winlen), 0) +
            np.expand_dims(np.arange(array.shape[0] + 1 - winlen), 0).T
    )

    return array[sub_windows[::stepsize]]


def batch_Y(df: pd.DataFrame, winlen: int, stepsize: int, num_relax: int,
            only_hits: bool = False, only_bounce: bool = False):
    if only_hits:
        df.replace(2, 0)
    elif only_bounce:
        df.replace(1, 0)

    #     df = df.fillna(0)

    array = df["hits"].to_numpy()

    window_Y = window(array, winlen, stepsize)

    start_i = winlen - num_relax

    if only_hits:

        # for two labels (nonhit and hit)
        labels = [1 if 1 in row[start_i:] else 0 for row in window_Y]

    elif only_bounce:
        # for two labels  (nonhit and bounce)
        labels = [1 if 2 in row[start_i:] else 0 for row in window_Y]

    else:
        # For three labels
        labels = [1 if 1 in row[start_i:] else (2 if 2 in row[start_i:] else 0) for row in window_Y]

    return np.array(labels)


def balance_dataset(all_X, all_Y):
    hit1 = np.count_nonzero(all_Y == 1)
    #     hit2  = np.count_nonzero(all_Y == 2)
    hit0 = np.count_nonzero(all_Y == 0)
    num_hit0_toretain = hit1

    ind_nohit = np.where(all_Y == 0)[0]
    np.random.shuffle(ind_nohit)
    selected_nohit = ind_nohit[:num_hit0_toretain]
    hits = np.where(all_Y != 0)[0]
    ind_toretain = np.concatenate((hits, selected_nohit))

    return all_X[ind_toretain], all_Y[ind_toretain]


def min_max_norm(x: pd.DataFrame, remove_key_points: bool):
    # Scaling x and y

    all_columns = x.columns.to_list()

    x_columns = all_columns[::2]
    y_columns = all_columns[1::2]

    x_df = x[x_columns]
    y_df = x[y_columns]

    # Scale by height and width of image
    # Divide selected columns by 1280
    x[x_columns] = x.loc[:, x_columns] / 1280

    # Divide selected columns by 720
    x[y_columns] = x.loc[:, y_columns] / 720

    return x


def horizontal_flip(df: pd.DataFrame, alpha: float):
    df_copy = df.copy()
    # det her skal laves om så
    # tar de først alpa % med af starten af clippet som horizontal flippet
    sample = df_copy.iloc[0: int(alpha * len(df_copy))]

    #     sample = df_copy.sample( int(alpha * len(df_copy) ))

    y = sample[["hits"]]
    x = sample.drop(columns=["hits"])

    flipped_df = _horizontal_flip(x)
    flipped_df["hits"] = y["hits"]
    # tror ikke den skal concatenate sammen, da det så er i samme clip. Der skal ligesom laves et nyt klip på en måde
    return flipped_df


def _horizontal_flip(x: pd.DataFrame):
    x_copy = x.copy()

    def modify_value(x):
        return 1280 - x

    for column in x.columns[1::2]:  # Iterate over every second column
        x_copy[column] = x_copy[column].apply(modify_value)

    return x_copy


def add_corners():
    path = '/kaggle/input/court-detection-coordinates/Dataset'
    all_csv = []
    games = glob.glob(path + "/*")

    for game in games:
        csv = glob.glob(game + "/*.csv")[0]
        all_csv.append(csv)

    return natsorted(all_csv)




def preprocess(df: pd.DataFrame, flip: bool, alpha: float, remove_key_points: bool, only_ball: bool = False, only_pose: bool = False):
    # each df is a df of clip.
    if flip:
        flipped_df = horizontal_flip(df, alpha)

        flipped_y = flipped_df[["hits"]]
        flipped_x = flipped_df.drop(columns=["hits", "Unnamed: 0"])

        if remove_key_points:
            flipped_x = flipped_x.drop(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                                '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'])
        if only_ball:
            col_to_remove = [str(i) for i in range(68)]
            flipped_x = flipped_x.drop(columns=col_to_remove)

        if only_pose:
            flipped_x = flipped_x.drop(columns=['ball_x', 'ball_y'])

        flipped_x = min_max_norm(flipped_x, remove_key_points)

    y = df[["hits"]]
    x = df.drop(columns=["hits", "Unnamed: 0"])

    if remove_key_points:
        x = x.drop(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            '34', '35', '36', '37', '38', '39', '40', '41', '42', '43'])
    if only_ball:
        col_to_remove = [str(i) for i in range(68)]
        x = x.drop(columns=col_to_remove)
        # x = x[['ball_x', 'ball_y']]

    if only_pose:
        x = x.drop(columns=['ball_x', 'ball_y'])

    x = min_max_norm(x, remove_key_points)

    if flip:
        return (x, y), (flipped_x, flipped_y)
    else:
        return (x, y), ('', '')


def batch_all_data(paths: List[str], winlen: int, stepsize: int,
                   num_relax: int, alpha: float, remove_key_points: bool, corners: bool,
                   only_hits: bool = False, only_bounce: bool = False, flip: bool = False,
                   only_ball: bool = False, only_pose: bool = False):
    num_features = len(pd.read_csv(paths[0]).columns) - 2

    nr_features_to_remove = 0

    if remove_key_points:
        nr_features_to_remove += 20
    if corners:
        nr_features_to_remove -= 8

    if only_pose:
        nr_features_to_remove += 2
    if only_ball:
        nr_features_to_remove += 68

    all_x = np.array([]).reshape((0, winlen, num_features - nr_features_to_remove))
    all_y = np.array([]).reshape((0,))

    num_matches = len(paths)

    for i in range(num_matches):
        df = pd.read_csv(paths[i])

        if corners:
            game = paths[i].split('/')[-3]
            clip = paths[i].split('/')[-2]

            corner_path = f'/kaggle/input/court-detection-coordinates/Dataset/{game}/{clip}/court.csv'

            corner_df = pd.read_csv(corner_path)

            # get the four corners
            corner_df = corner_df[corner_df['point'].isin([0, 1, 2, 3])][['x-coordinate', 'y-coordinate']]
            corner_list = corner_df.values.flatten()

            # Generating unique column names based on the length of the values_to_add list
            new_columns = [f'corner_{i}' for i in range(len(corner_list))]

            # Adding each item in the list to its own column for each row
            df = df.assign(**dict(zip(new_columns, corner_list)))

        if flip:
            (x, y), (flipped_x, flipped_y) = preprocess(df, flip=True, alpha=alpha, remove_key_points=remove_key_points,
                                                        only_ball=only_ball,
                                                        only_pose=only_pose)  # Set flip = True for flipping

        else:
            (x, y), _ = preprocess(df, flip=False, alpha=alpha, remove_key_points=remove_key_points,
                                   only_ball=only_ball, only_pose=only_pose)

        # batch the data
        batch_x = batch_X(x, winlen, stepsize)
        if flip:
            flipped_batch_x = batch_X(flipped_x, winlen, stepsize)
            batch_x = np.concatenate((batch_x, flipped_batch_x))

        if only_hits:
            batch_y = batch_Y(y, winlen, stepsize, num_relax, only_hits=True)
            if flip:
                flipped_batch_y = batch_Y(flipped_y, winlen, stepsize, num_relax, only_hits=True)
                batch_y = np.concatenate((batch_y, flipped_batch_y))

        elif only_bounce:
            batch_y = batch_Y(y, winlen, stepsize, num_relax, only_bounce=True)
            if flip:
                flipped_batch_y = batch_Y(flipped_y, winlen, stepsize, num_relax, only_bounce=True)
                batch_y = np.concatenate((batch_y, flipped_batch_y))

        else:
            batch_y = batch_Y(y, winlen, stepsize, num_relax)
            if flip:
                flipped_batch_y = batch_Y(flipped_y, winlen, stepsize, num_relax)
                batch_y = np.concatenate((batch_y, flipped_batch_y))

        all_x = np.concatenate((all_x, batch_x))
        all_y = np.concatenate((all_y, batch_y))
    return (all_x, all_y)

def dataBatcher(winlen: int, n_split: int, stepsize: int, num_relax: int,
                balance: bool, alpha: float, remove_key_points: bool, corners: bool,
                only_hits: bool = False, only_bounce: bool = False, flip: bool = False):
    data_paths = get_all_csv_paths()

    random.seed(0)
    random.shuffle(data_paths)

    paths_dict = dict()

    test_games = ['game1', 'game8']

    test_paths = []
    _train_paths = []

    for path in data_paths:
        game_name = path.split('/')[-3]
        clip_name = path.split('/')[-2]

        if game_name in test_games:
            test_paths.append(path)
        # Remove game11
        elif game_name == 'game11':
            continue
        # game 7 clip 7 is alreaddy removed in filtered poses
        else:
            _train_paths.append(path)

    split_amount = int(len(_train_paths) * n_split)
    val_paths = _train_paths[split_amount:]
    train_paths = _train_paths[: split_amount]

    paths_dict['Validation'] = val_paths
    paths_dict['Train'] = train_paths
    paths_dict['Test'] = test_paths

    # Create train data
    x_train, y_train = batch_all_data(train_paths, winlen, stepsize, num_relax,
                                      alpha=alpha, remove_key_points=remove_key_points,
                                      corners=corners, only_hits=only_hits, only_bounce=only_bounce,
                                      flip=flip)
    # Create validation data
    x_val, y_val = batch_all_data(val_paths, winlen, stepsize, num_relax,
                                  alpha=alpha, remove_key_points=remove_key_points,
                                  corners=corners, only_hits=only_hits, only_bounce=only_bounce, flip=False)
    # Create test data
    x_test, y_test = batch_all_data(test_paths, winlen, stepsize, num_relax,
                                    alpha=alpha, remove_key_points=remove_key_points,
                                    corners=corners, only_hits=only_hits, only_bounce=only_bounce, flip=False)

    if balance:
        x_train, y_train = balance_dataset(x_train, y_train)

    # generates a random permutation index
    perm_train = np.random.permutation(len(x_train))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]

    perm_val = np.random.permutation(len(x_val))
    x_val = x_val[perm_val]
    y_val = y_val[perm_val]

    # perm_test = np.random.permutation(len(x_test))
    # x_test = x_test[perm_test]
    # y_test = y_test[perm_test]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), paths_dict


# new custom eval metrics
def find_components(lst):
    components = []
    current_component = None
    component_indices = []

    for i, val in enumerate(lst):
        if val == 1:
            if val != current_component:
                if current_component is not None:
                    components.append(component_indices)
                current_component = val
                component_indices = [i]
            else:
                component_indices.append(i)
        else:
            if current_component == 1:
                components.append(component_indices)
                current_component = None
                component_indices = []

    # Check if the last component is 1
    if current_component == 1:
        components.append(component_indices)

    return components


def find_component_intersections(components_a, components_b):
    intersections = []
    for component_a in components_a:
        for component_b in components_b:
            intersection = set(component_a).intersection(component_b)
            if intersection:
                intersections.append((component_a, component_b, list(intersection)))
    return intersections


def custom_evaluation_hits(y_true, preds):
    y_components = find_components(y_true)
    y_new_components = y_components.copy()

    preds_components = find_components(preds)
    preds_new_components = preds_components.copy()

    intersections = find_component_intersections(y_components, preds_components)

    overlaps = []
    #     print(intersections)

    for intersection in intersections:
        #         print(intersection)
        # find missing hits
        if intersection[0] in y_new_components:
            y_new_components.remove(intersection[0])

        if intersection[1] in preds_new_components:
            preds_new_components.remove(intersection[1])

        # find wrong hits
        #         preds_new_components.remove(intersection[1])

        # find overlap
        _intersection = len(intersection[2])
        _union = len(set(intersection[0] + intersection[1]))
        overlaps.append(_intersection / _union)

    frac_missing_hits = len(y_new_components) / len(y_components)

    if len(preds_components):
        frac_wrong_hits = len(preds_new_components) / len(preds_components)
    else:
        frac_wrong_hits = 1

    if not overlaps:
        overlaps = 0

    return frac_missing_hits, frac_wrong_hits, np.mean(overlaps)