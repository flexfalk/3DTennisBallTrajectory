import torch
from Utils.Reconstruction3D.Reconstruction3D_utils import get_court_dimension, create_3d_trajectory, create_3d_trajectory_with_spin, get_important_cam_params, project_points_numpy
import numpy as np
import cv2

def check_if_land_outside(trajectory):
    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    landing_coordinate = trajectory[-1][0:2]

    return (-1 * half_court_width - 1 < landing_coordinate[0] < half_court_width + 1) and (
                -1 * half_court_length - 1 < landing_coordinate[1] < half_court_length + 1)


def check_if_crosses_net(trajectory):
    start_coordinate_y = trajectory[0][1]

    landing_coordinate_y = trajectory[-1][1]

    return 0 > start_coordinate_y * landing_coordinate_y


def custom_list_comp(compressed_list):
    return [el[0] for el in compressed_list]


def create_synthetic_dataset(random_number, cam_params, df, spin=False):
    df["game"] = cam_params.iloc[random_number]["game"].to_list()
    df["clip"] = cam_params.iloc[random_number]["clip"].to_list()

    if spin:
        df["starting_params"] = df.apply(
            lambda row: [row['x'], row['y'], row['z'], row['vx'], row['vy'], row['vz'], row['wx'], row['wy'],
                         row['wz']], axis=1)
        df["trajectory3D"] = df.apply(lambda row: create_3d_trajectory_with_spin(
            torch.tensor([row["starting_params"]], dtype=torch.float).reshape(-1, 1).T, 15, True), axis=1)
    else:
        df["starting_params"] = df.apply(lambda row: [row['x'], row['y'], row['z'], row['vx'], row['vy'], row['vz']],
                                         axis=1)
        df["trajectory3D"] = df.apply(
            lambda row: create_3d_trajectory(torch.tensor([row["starting_params"]], dtype=torch.float).reshape(-1, 1).T,
                                             15, True), axis=1)

    _projection = []

    for i in range(len(df)):
        row = df.iloc[i]
        game = row["game"]
        clip = row["clip"]

        homography, rotation_matrix, tvecs, cam_mtx, dist = get_important_cam_params(game, clip, cam_params)

        project = project_points_numpy(row["trajectory3D"], torch.tensor(rotation_matrix), tvecs, cam_mtx, dist)
        _projection.append(project)

    df["projection"] = _projection

    # filter on land outsidecourt
    df = df[df["trajectory3D"].apply(lambda row: check_if_land_outside(row).item())]

    # filter on cross net
    df = df[df["trajectory3D"].apply(lambda row: check_if_crosses_net(row).item())]

    # filter x axis outside projection
    df = df[df["projection"].apply(lambda row: (row[-1][0] > 0) and (row[-1][0] < 1280))]

    # filter y axis outside projection
    df = df[df["projection"].apply(lambda row: (row[-1][1] > 0) and (row[-1][1] < 720))]
    df["trajectory3D"] = df["trajectory3D"].apply(lambda x: x.tolist())
    df["projection"] = df["projection"].apply(lambda x: x.tolist())

    return df


def create_synthetic_dataset_bigger_and_better(random_number, cam_params_train, df, spin=False):
    df["game"] = cam_params_train.iloc[random_number]["game"].to_list()
    df["clip"] = cam_params_train.iloc[random_number]["clip"].to_list()

    if spin:
        df["starting_params"] = df.apply(
            lambda row: [row['x'], row['y'], row['z'], row['vx'], row['vy'], row['vz'], row['wx'], row['wy'],
                         row['wz']], axis=1)
        df["trajectory3D"] = df.apply(lambda row: create_3d_trajectory_with_spin(
            torch.tensor([row["starting_params"]], dtype=torch.float).reshape(-1, 1).T, 15, True), axis=1)

    else:
        df["starting_params"] = df.apply(lambda row: [row['x'], row['y'], row['z'], row['vx'], row['vy'], row['vz']],
                                         axis=1)
        df["trajectory3D"] = df.apply(
            lambda row: create_3d_trajectory(torch.tensor([row["starting_params"]], dtype=torch.float).reshape(-1, 1).T,
                                             15, True), axis=1)

    _projection = []

    trans_vector = np.array(cam_params_train["translation_vector"].apply(lambda row: custom_list_comp(row)).tolist())

    means_t = np.mean(trans_vector, axis=0)
    stds_t = np.std(trans_vector, axis=0)

    rotation_vector = np.array(cam_params_train["rotation_vector"].apply(lambda row: custom_list_comp(row)).tolist())

    means_r = np.mean(rotation_vector, axis=0)
    stds_r = np.std(rotation_vector, axis=0)

    for i in range(len(df)):
        row = df.iloc[i]
        game = row["game"]
        clip = row["clip"]

        cam_parameters = cam_params_train[(cam_params_train["game"] == game) & (cam_params_train["clip"] == clip)].iloc[0]

        cam_mtx = np.array(cam_parameters["camera_matrix"])

        dist = np.array(cam_parameters["dist"])

        random_tvec = np.random.normal(means_t, stds_t).reshape(3, 1)

        random_rvec = np.random.normal(means_r, stds_r).reshape(3, 1)

        rotation_matrix, _ = cv2.Rodrigues(random_rvec)

        project = project_points_numpy(row["trajectory3D"], torch.tensor(rotation_matrix), random_tvec, cam_mtx, dist)
        _projection.append(project)

    df["projection"] = _projection

    # filter on land outsidecourt
    df = df[df["trajectory3D"].apply(lambda row: check_if_land_outside(row).item())]

    # filter on cross net
    df = df[df["trajectory3D"].apply(lambda row: check_if_crosses_net(row).item())]

    # filter x axis outside projection
    df = df[df["projection"].apply(lambda row: (row[-1][0] > 0) and (row[-1][0] < 1280))]

    # filter y axis outside projection
    df = df[df["projection"].apply(lambda row: (row[-1][1] > 0) and (row[-1][1] < 720))]

    # # df
    df["trajectory3D"] = df["trajectory3D"].apply(lambda x: x.tolist())
    df["projection"] = df["projection"].apply(lambda x: x.tolist())

    return df