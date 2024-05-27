import pandas as pd
import ast
import torch
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

def parse_list(string):
    try:
        # Use ast.literal_eval to safely parse the string representation of a list
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        # If parsing fails, return None or any other default value as needed
        return None


# Custom collate function to handle sequences of varying lengths
def custom_collate(batch):
    for i in batch:
        if i is None:
            return None
    #     print(mtx)
    inputs, mtx, dist, rvecs, tvecs, rotation_matrix, poses, player1_homography, player2_homography, shot_id = zip(
        *batch)
    #     print(torch.stack(mtx))

    padded_inputs = [torch.nn.functional.pad(seq, (0, 50 - len(seq[1])), value=-1) for seq in inputs]
    padded_poses = [torch.nn.functional.pad(seq, (0, 0, 0, 50 - len(seq[:, 1])), value=-1, mode='constant') for seq in
                    poses]

    padded_inputs = torch.stack(padded_inputs)
    # padded_poses = torch.stack(padded_poses)

    return padded_inputs, torch.stack(mtx), torch.stack(dist), torch.stack(rvecs), torch.stack(tvecs), torch.stack(
        rotation_matrix), padded_poses, player1_homography, player2_homography, shot_id


def create_3d_trajectory(output, N, until_ground_hit=False, stepsize=3):
    position = output[:, 0:3]
    v = output[:, 3:6]
    fps = 30
    D = 0.00114
    #     D = output[:, -1]
    m = 0.056
    t = 0  # Start time

    N_max = stepsize * N  # How many points we want per frame
    t_max = N / fps
    delta_t = t_max / N_max  # Time interval

    g = torch.tensor([0.0, 0.0, -9.81], device=output.device)

    # positions = torch.tensor(position, device=output.device).view(1, 3)
    # positions = torch.tensor(position, device=output.device).clone().view(1, 3)
    positions = position.clone().view(1, 3)

    if until_ground_hit:
        while position[0, 2] > 0:
            v_norm = torch.norm(v)

            a = g - 1 * (D / m) * v_norm * v
            v = v + a * delta_t
            position = position + v * delta_t + 0.5 * a * delta_t ** 2
            positions = torch.cat((positions, position.view(1, 3)), dim=0)

    else:
        for i in range(N_max - 1):
            v_norm = torch.norm(v)

            a = g - 1 * (D / m) * v_norm * v
            v = v + a * delta_t
            position = position + v * delta_t + 0.5 * a * delta_t ** 2
            positions = torch.cat((positions, position.view(1, 3)), dim=0)

    indices = [i for i in range(0, len(positions), 3)]
    indexed_positions = positions[indices]

    return indexed_positions


def create_3d_trajectory_with_spin(output, N, until_ground_hit=False, stepsize=3):
    position = output[:, 0:3]
    v = output[:, 3:6]
    fps = 30
    spin = output[:, 6:9]

    D = 0.00114
    #     D = output[:, -1]
    m = 0.056
    t = 0  # Start time
    N_max = stepsize * N  # How many points we want per frame
    t_max = N / fps
    delta_t = t_max / N_max  # Time interval

    g = torch.tensor([0.0, 0.0, -9.81], device=output.device)

    magnus_coefficient = 0.00041

    # positions = torch.tensor(position, device=output.device).view(1, 3)
    # positions = torch.tensor(position, device=output.device).clone().view(1, 3)
    positions = position.clone().view(1, 3)

    if until_ground_hit:
        while position[0, 2] > 0:
            v_norm = torch.norm(v)

            magnus_force = magnus_coefficient * torch.cross(spin, v)

            drag_force = - 1 * (D / m) * v_norm * v

            a = g + drag_force + magnus_force #/ m

            v = v + a * delta_t
            position = position + v * delta_t + 0.5 * a * delta_t ** 2
            positions = torch.cat((positions, position.view(1, 3)), dim=0)

    else:
        for i in range(N_max - 1):
            v_norm = torch.norm(v)

            magnus_force = magnus_coefficient * torch.cross(spin, v)

            drag_force = - 1 * (D / m) * v_norm * v

            a = g + drag_force + magnus_force #/ m

            v = v + a * delta_t
            position = position + v * delta_t + 0.5 * a * delta_t ** 2
            positions = torch.cat((positions, position.view(1, 3)), dim=0)

    indices = [i for i in range(0, len(positions), 3)]
    indexed_positions = positions[indices]

    return indexed_positions

def project_points_torch(trajectory: torch.tensor, rotation_matrix, translation_vector, camera_matrix,
                         distortion_coeffs):
    # Convert inputs to double precision
    trajectory = trajectory.double()
    camera_matrix = camera_matrix.double()
    distortion_coeffs = distortion_coeffs[0]

    num_rows = trajectory.size(0)
    # Create a column tensor of ones with the same number of rows as the original tensor
    ones_column = torch.ones(num_rows, 1, device=trajectory.device)

    # Concatenate the original tensor and the ones column along the second dimension (columns)
    trajectory = torch.cat((trajectory, ones_column), dim=1)

    rot_and_trans = torch.cat((rotation_matrix.T, translation_vector.T)).T

    foo = torch.matmul(rot_and_trans, trajectory.T).T

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    f = torch.tensor([fx, fy], device=trajectory.device)
    c = torch.tensor([cx, cy], device=trajectory.device)

    x_and_y = foo[:, 0:2]
    z = foo[:, 2]
    z = z.view(z.shape[0], 1)

    z = torch.cat((z, z), dim=1)

    points_2d = f * x_and_y / z + c

    x1_and_y1 = x_and_y / z

    #     N = x1_and_y1.shape[0]
    r2 = x1_and_y1[:, 0] ** 2 + x1_and_y1[:, 1] ** 2

    r4 = r2 ** 2
    r6 = r2 ** 3
    radial = 1.0 + distortion_coeffs[0] * r2 + distortion_coeffs[1] * r4 + distortion_coeffs[4] * r6

    tangential_x = 2.0 * distortion_coeffs[2] * x1_and_y1[:, 0] * x1_and_y1[:, 1] + distortion_coeffs[3] * (
                r2 + 2.0 * x1_and_y1[:, 0] ** 2)

    tangential_y = distortion_coeffs[2] * (r2 + 2.0 * x1_and_y1[:, 1] ** 2) + 2.0 * distortion_coeffs[3] * x1_and_y1[:,
                                                                                                           0] * x1_and_y1[
                                                                                                                :, 1]

    x2 = x1_and_y1[:, 0] * radial + tangential_x
    y2 = x1_and_y1[:, 1] * radial + tangential_y

    bro = torch.cat((x2.view(x2.shape[0], 1), y2.view(y2.shape[0], 1)), dim=1)

    u_and_v = f * bro + c

    return u_and_v.float()


def get_court_dimension():
    court_length = 23.77
    court_width = 10.97
    half_court_length = court_length / 2
    half_court_width = court_width / 2
    net_height_middle = 0.91
    net_height_sides = 1.067

    return court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides


def create_synthetic_shots(N: int):

    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    N_front_players_shots = int(N / 2)
    N_back_player_shots = int(N / 2)

    front_player_shots = []
    back_player_shots = []



    for i in range(N_front_players_shots):
        x = np.random.uniform(-half_court_width - 1, half_court_width + 1)
        y = np.random.uniform(-half_court_length - 1, - 1)
        z = np.random.uniform(0.1, 3)

        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(10, 30)
        vz = np.random.uniform(0, 5)

        init_params = [x, y, z, vx, vy, vz]
        front_player_shots.append(init_params)

    for i in range(N_back_player_shots):
        x = np.random.uniform(-half_court_width - 1, half_court_width + 1)
        y = np.random.uniform(1, half_court_length + 1)
        z = np.random.uniform(0.1, 3)

        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(-10, -30)
        vz = np.random.uniform(0, 5)

        init_params = [x, y, z, vx, vy, vz]
        back_player_shots.append(init_params)

    return front_player_shots + back_player_shots
import math
def create_synthetic_shots_with_spin(N: int):

    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    N_front_players_shots = int(N / 2)
    N_back_player_shots = int(N / 2)

    front_player_shots = []
    back_player_shots = []



    for i in range(N_front_players_shots):
        x = np.random.uniform(-half_court_width - 1, half_court_width + 1)
        y = np.random.uniform(-half_court_length - 1, - 1)
        z = np.random.uniform(0.1, 3)

        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(10, 30)
        vz = np.random.uniform(0, 5)

        wx = np.random.uniform(-10*2*math.pi, 10*2*math.pi)
        wy = np.random.uniform(-1*2*math.pi, 1*2*math.pi)
        wz = np.random.uniform(-5*2*math.pi, 5*2*math.pi)

        init_params = [x, y, z, vx, vy, vz, wx, wy, wz]
        front_player_shots.append(init_params)

    for i in range(N_back_player_shots):
        x = np.random.uniform(-half_court_width - 1, half_court_width + 1)
        y = np.random.uniform(1, half_court_length + 1)
        z = np.random.uniform(0.1, 3)

        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(-10, -30)
        vz = np.random.uniform(0, 5)

        wx = np.random.uniform(-10*2*math.pi, 10*2*math.pi)
        wy = np.random.uniform(-1*2*math.pi, 1*2*math.pi)
        wz = np.random.uniform(-5*2*math.pi, 5*2*math.pi)

        init_params = [x, y, z, vx, vy, vz, wx, wy, wz]

        back_player_shots.append(init_params)

    return front_player_shots + back_player_shots


def project_points_numpy(trajectory, rotation_matrix, translation_vector, camera_matrix, distortion_coeffs):
    # Convert inputs to double precision
    trajectory = np.array(trajectory, dtype=np.float64)
    camera_matrix = np.array(camera_matrix, dtype=np.float64)
    distortion_coeffs = np.array(distortion_coeffs[0], dtype=np.float64)

    num_rows = trajectory.shape[0]
    # Create a column matrix of ones with the same number of rows as the original matrix
    ones_column = np.ones((num_rows, 1), dtype=np.float64)

    # Concatenate the original matrix and the ones column along the second dimension (columns)
    trajectory = np.concatenate((trajectory, ones_column), axis=1)

    rot_and_trans = np.concatenate((rotation_matrix, translation_vector), axis=1)
    #     print(rot_and_trans.shape)
    foo = np.matmul(rot_and_trans, trajectory.T).T

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    f = np.array([fx, fy], dtype=np.float64)
    c = np.array([cx, cy], dtype=np.float64)

    x_and_y = foo[:, 0:2]
    z = foo[:, 2].reshape(-1, 1)

    z = np.concatenate((z, z), axis=1)

    points_2d = f * x_and_y / z + c

    x1_and_y1 = x_and_y / z

    r2 = x1_and_y1[:, 0] ** 2 + x1_and_y1[:, 1] ** 2

    r4 = r2 ** 2
    r6 = r2 ** 3
    radial = 1.0 + distortion_coeffs[0] * r2 + distortion_coeffs[1] * r4 + distortion_coeffs[4] * r6

    tangential_x = 2.0 * distortion_coeffs[2] * x1_and_y1[:, 0] * x1_and_y1[:, 1] + distortion_coeffs[3] * (
                r2 + 2.0 * x1_and_y1[:, 0] ** 2)

    tangential_y = distortion_coeffs[2] * (r2 + 2.0 * x1_and_y1[:, 1] ** 2) + 2.0 * distortion_coeffs[3] * x1_and_y1[:,
                                                                                                           0] * x1_and_y1[
                                                                                                                :, 1]

    x2 = x1_and_y1[:, 0] * radial + tangential_x
    y2 = x1_and_y1[:, 1] * radial + tangential_y

    bro = np.concatenate((x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)

    u_and_v = f * bro + c

    return u_and_v.astype(np.float32)



def project_single_point_numpy(point, rotation_matrix, translation_vector, camera_matrix, distortion_coeffs):
    # Convert inputs to double precision
    point = np.array(point, dtype=np.float64)
    camera_matrix = np.array(camera_matrix, dtype=np.float64)

    distortion_coeffs = np.array(distortion_coeffs[0], dtype=np.float64)

    # Add homogeneous coordinate (w = 1)
    point = np.append(point, 1)

    # Concatenate rotation matrix and translation vector
    rot_and_trans = np.concatenate((rotation_matrix, translation_vector), axis=1)

    # Project 3D point into 2D image plane
    point_homogeneous = np.matmul(rot_and_trans, point)
    x, y, z = point_homogeneous

    # Intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Normalize
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy

    # Distortion correction
    r2 = (x / z) ** 2 + (y / z) ** 2
    radial = 1.0 + distortion_coeffs[0] * r2 + distortion_coeffs[1] * r2 ** 2 + distortion_coeffs[4] * r2 ** 3
    tangential_x = 2.0 * distortion_coeffs[2] * (x / z) * (y / z) + distortion_coeffs[3] * (r2 + 2.0 * (x / z) ** 2)
    tangential_y = distortion_coeffs[2] * (r2 + 2.0 * (y / z) ** 2) + 2.0 * distortion_coeffs[3] * (x / z) * (y / z)

    # Apply distortion correction
    x_corrected = (x / z) * radial + tangential_x
    y_corrected = (y / z) * radial + tangential_y

    # Final pixel coordinates
    u_corrected = (fx * x_corrected) + cx
    v_corrected = (fy * y_corrected) + cy

    return np.array([u_corrected, v_corrected], dtype=np.float32)


def mse(vector1, vector2):
    """
    Calculate the Mean Squared Error (MSE) between two vectors of size N x 2.

    Args:
    - vector1: First vector of size N x 2.
    - vector2: Second vector of size N x 2.

    Returns:
    - mse: Mean Squared Error between the two vectors.
    """
    # Compute squared differences
    squared_diff = np.square(vector1 - vector2)

    # Calculate mean squared error
    mse = np.mean(squared_diff)

    rmse = np.sqrt(mse)

    return rmse


def error_distance_landing(pred_traj, true_traj, homography_matrix):
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu()
    if isinstance(true_traj, torch.Tensor):
        true_traj = true_traj.cpu()
    if isinstance(homography_matrix, torch.Tensor):
        homography_matrix = homography_matrix.cpu()
    # pred_position = np.array(pred_traj)[-1, :2]  # .reshape((-1,1,2))[-1]

    pred_position = np.array(pred_traj)[0, -1]
    for i in range(len(pred_traj)):
        if pred_traj[i, -1] < 0:
            break
        else:
            pred_position = np.array(pred_traj[i, 0:2])

    true_position = np.array(true_traj[0, -1]).reshape(-1, 1, 2)  # .reshape((-1,1,2))

    # Find homography for true 2d image coordinates
    true_position = cv2.perspectiveTransform(true_position, np.array(homography_matrix)).squeeze()

    dist = np.linalg.norm(true_position - pred_position)

    return dist


def ball_hits_court(pred_traj, true_traj, homography_matrix):
    """
    This function dvivides the court in 12 square, three in width and 4 in height, meaning each half of the court consists of 6 squares.
    The squares are numbered from top left going right.

    pred_traj : the predicted 3D trajectory
    true_traj : the true 2D trajectory
    homography_matrix : the homography matrix for the given shot
    return : a number determining which square of the field the true and predicted ball lands in
    """

    if isinstance(pred_traj, torch.Tensor):
        pred_traj = np.array(pred_traj.cpu())
    if isinstance(true_traj, torch.Tensor):
        true_traj = true_traj.cpu()
    if isinstance(homography_matrix, torch.Tensor):
        homography_matrix = homography_matrix.cpu()

    pred_position = np.array(pred_traj)[0, 0:2]
    for i in range(len(pred_traj)):
        if pred_traj[i, -1] < 0:
            break
        else:
            pred_position = np.array(pred_traj[i, 0:2])

    true_position = np.array(true_traj[0, -1]).reshape(-1, 1, 2)  # .reshape((-1,1,2))

    # Find homography for true 2d image coordinates
    true_position = cv2.perspectiveTransform(true_position, np.array(homography_matrix)).squeeze()

    # Define the boundaries of each square in real-world coordinates for back and front court
    court_length = 23.77
    half_length = court_length / 2
    court_width = 10.97
    half_width = court_width / 2
    pred_square = -1
    true_square = -1

    boundaries_width = [-half_width, -half_width / 2, half_width / 2, half_width]
    boundaries_length = [half_length, half_length / 2, 0, -half_length / 2, -half_length]

    for i in range(len(boundaries_width) - 1):
        if boundaries_width[i] < pred_position[0] <= boundaries_width[i + 1]:
            for j in range(len(boundaries_length) - 1):
                if boundaries_length[j] > pred_position[1] >= boundaries_length[j + 1]:
                    pred_square = (j * 3) + i
                    break

    for i in range(len(boundaries_width) - 1):
        if boundaries_width[i] < true_position[0] <= boundaries_width[i + 1]:
            for j in range(len(boundaries_length) - 1):
                if boundaries_length[j] > true_position[1] >= boundaries_length[j + 1]:
                    true_square = (j * 3) + i
                    break

    return pred_square, true_square



def get_important_cam_params(game, clip, camera_params_train):
    # find camera parameters
    cam_parameters = \
    camera_params_train[(camera_params_train["game"] == game) & (camera_params_train["clip"] == clip)].iloc[0]
    cam_mtx = np.array(cam_parameters["camera_matrix"])
    rvecs = np.array(cam_parameters["rotation_vector"])
    rvecs = rvecs.reshape(3, 1)
    tvecs = np.array(cam_parameters["translation_vector"])
    tvecs = tvecs.reshape(3, 1)
    dist = np.array(cam_parameters["dist"])

    rotation_matrix, _ = cv2.Rodrigues(rvecs)

    rot_and_trans = np.concatenate((rotation_matrix, tvecs), axis=1)
    # rot_and_trans.shape

    camProj = cam_mtx @ rot_and_trans
    homography = torch.tensor(eval(cam_parameters["homography_matrix"]), dtype=torch.float64)

    return homography, rotation_matrix, tvecs, cam_mtx, dist

def calculate_accuracy(group):
    return accuracy_score(group['true_tiles'], group['predicted_tiles'])

# Function to calculate F1 macro score
def calculate_f1_macro(group):
    return f1_score(group['true_tiles'], group['predicted_tiles'], average='macro')


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def average_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same number of vectors.")

    total_distance = 0
    for vec1, vec2 in zip(list1, list2):
        total_distance += euclidean_distance(vec1, vec2)

    average_dist = total_distance / len(list1)
    return average_dist


def find_closest_player(player1, player2, points_2d):
    player1_hip = player1[:, 14:16]
    player2_hip = player2[:, 14:16]

    start_2d = points_2d[0]
    end_2d = points_2d[-1]

    distance_player1 = np.linalg.norm(start_2d - player1_hip[0])
    distance_player2 = np.linalg.norm(start_2d - player2_hip[0])

    closest_player = np.argmin([distance_player1, distance_player2])

    return [player1, player2][closest_player]


def get_players(pose_numpy, start):
    pose_df = pose_numpy[start, :]

    player1 = pose_df[0:34].reshape(1, 34)

    player2 = pose_df[34:].reshape(1, 34)

    return player1, player2


def fix_trajectory(traj):
    if traj[0] <= 0:
        vals_to_change = []
        i = 0

        while traj[i] <= 0:
            vals_to_change.append(i)
            i += 1

        for j in vals_to_change:
            traj[j] = traj[i]

    for i in range(len(traj)):
        if traj[i] <= 0:
            traj[i] = traj[i - 1]

    return traj


def clean_true(df):
    v_to_remove = ["nan", " ", " nan"]
    df["x_true"] = df.apply(
        lambda row: [float(v) if v not in v_to_remove else 0 for v in row["x_true"].strip("[]").split(", ")], axis=1)
    df["y_true"] = df.apply(
        lambda row: [float(v) if v not in v_to_remove else 0 for v in row["y_true"].strip("[]").split(", ")], axis=1)

    df['x_true'] = df['x_true'].apply(lambda row: fix_trajectory(row))
    df['y_true'] = df['y_true'].apply(lambda row: fix_trajectory(row))
    df = df.reset_index()
    df = df.drop([432, 131, 483, 233, 440, 6, 420, 34, 317, 451, 272, 438, 342, 430])
    df = df[df['x_true'].apply(lambda row: len(row) > 5)]
    return df.reset_index()


def clean_predicted(ball_df):
    ball_df = ball_df[ball_df["x_WASB"].apply(lambda x: len(x) > 3)].reset_index()
    ball_df = ball_df.drop([293, 141, 462, 460, 443, 471, 35, 364, 6])
    ball_df['x_WASB'] = ball_df['x_WASB'].apply(lambda row: str(fix_trajectory(eval(row))))
    ball_df['y_WASB'] = ball_df['y_WASB'].apply(lambda row: str(fix_trajectory(eval(row))))
    ball_df = ball_df[ball_df['x_WASB'].apply(lambda row: len(eval(row)) > 5)]
    return ball_df.reset_index()


def find_homography(closest_player, game, clip, camera_params_train):
    homography, rotation_matrix, tvecs, cam_mtx, dist = get_important_cam_params(game, clip, camera_params_train)

    _left_foot = closest_player[:, 30:32]
    _right_foot = closest_player[:, 32:34]

    closest_player_mid = np.array(
        ((_left_foot[:, 0] + _right_foot[:, 0]) / 2, (_left_foot[:, 1] + _right_foot[:, 1]) / 2)).T

    # hip = closest_player[:, 14:16]
    location_3d = np.array([closest_player_mid[0][0], closest_player_mid[0][1]]).reshape(-1, 1, 2)

    # Find homography for true 2d image coordinates
    player_homography = cv2.perspectiveTransform(location_3d, np.array(homography)).squeeze()

    return player_homography
import scipy.optimize
from sklearn.metrics import mean_squared_error

def solve_trajectory(shot_number, camera_params_train, ball_df, pose_numpy, ground_truth=True, spin=False,
                     priors=False):
    # find shot information
    game = ball_df.iloc[shot_number]["game"]
    clip = ball_df.iloc[shot_number]["clip"]

    start = ball_df.iloc[shot_number]["start"]
    end = ball_df.iloc[shot_number]["end"]

    homography, rotation_matrix, tvecs, cam_mtx, dist = get_important_cam_params(game, clip, camera_params_train)
    # find data

    if ground_truth:
        trajx = np.array(ball_df.iloc[shot_number]["x_true"])
        trajy = np.array(ball_df.iloc[shot_number]["y_true"])
    else:
        trajx = np.array(eval(ball_df.iloc[shot_number]["x_WASB"]))
        trajy = np.array(eval(ball_df.iloc[shot_number]["y_WASB"]))

    trajx = trajx.reshape(len(trajx), 1)
    trajy = trajy.reshape(len(trajx), 1)
    traj = np.concatenate((trajx, trajy), axis=1)

    xi = traj[0, :]
    xf = traj[-1, :]

    loss_list = []

    x_scale = 10.97 / 2
    y_scale = 23.77 / 2

    fps = 30
    substeps = 10
    N = len(traj)

    if spin:
        bounds = [(-x_scale - 2, x_scale + 2), (-y_scale - 2, y_scale + 2), (0.0, 1.5), (-20, 20), (-40, 40), (-5, 10),
                  (-10 * 2 * math.pi, 10 * 2 * math.pi), (-1 * 2 * math.pi, 1 * 2 * math.pi),
                  (-5 * 2 * math.pi, 5 * 2 * math.pi)]
        initg = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    else:
        bounds = [(-x_scale - 2, x_scale + 2), (-y_scale - 2, y_scale + 2), (0.0, 1.5), (-20, 20), (-40, 40),
                  (-5, 10)]  # , (0, 1)
        initg = [1, 1, 1, 1, 1, 1]

    def f_spin(p):

        position_initial = np.array(p[:3])
        position = np.array(p[:3])
        vel = np.array(p[3:6])
        spin = np.array(p[6:9])
        fps = 30
        D = 0.0012
        m = 0.056
        g = [0.0, 0.0, -9.82]
        magnus_coefficient = 0.00041

        N_max = N * 3
        t_max = N / fps
        dt = t_max / N_max

        positions = []

        pixel_err = 0
        tid = 0

        for t in range(N_max - 1):

            if t % 3 == 0:
                new_position = project_single_point_numpy(position, rotation_matrix, tvecs, cam_mtx, dist)
                positions.append(new_position)

                tid += 1
            v_norm = np.linalg.norm(vel)
            magnus_force = magnus_coefficient * np.cross(spin, vel)

            drag_force = - 1 * (D / m) * v_norm * vel

            a = g + drag_force + magnus_force #/ m
            vel = vel + a * dt
            position = position + vel * dt + 0.05 * a * dt ** 2
        #         loss = np.linalg.norm(positions- traj)
        loss = mean_squared_error(positions, traj)

        player1, player2 = get_players(pose_numpy, start)
        closest_player = find_closest_player(np.array(player1), np.array(player2), traj)

        homography_player = find_homography(closest_player, game, clip, camera_params_train)
        homography_player = np.array([homography_player[0], homography_player[1], 1.5])

        player_position_loss = np.linalg.norm(homography_player - position_initial) ** 2

        if priors:
            all_loss = loss + player_position_loss
        else:
            all_loss = loss

        loss_list.append(loss)

        return all_loss

    def f_ours(p):

        position_initial = np.array(p[:3])
        position = np.array(p[:3])
        vel = np.array(p[3:6])
        #fps = 30
        D = 0.0012
        m = 0.056
        g = [0.0, 0.0, -9.82]
        fps = 30
        N_max = N * 3
        t_max = N / fps
        dt = t_max / N_max

        positions = []

        pixel_err = 0
        tid = 0

        for t in range(N_max - 1):

            if t % 3 == 0:
                new_position = project_single_point_numpy(position, rotation_matrix, tvecs, cam_mtx, dist)
                positions.append(new_position)

                tid += 1
            v_norm = np.linalg.norm(vel)
            a = g - 1 * (D / m) * v_norm * vel
            vel = vel + a * dt
            position = position + vel * dt + 0.05 * a * dt ** 2

        loss = mean_squared_error(positions, traj)
        #         loss = np.linalg.norm(positions- traj)

        player1, player2 = get_players(pose_numpy, start)
        closest_player = find_closest_player(np.array(player1), np.array(player2), traj)

        homography_player = find_homography(closest_player, game, clip, camera_params_train)
        homography_player = np.array([homography_player[0], homography_player[1], 1.5])

        player_position_loss = np.linalg.norm(homography_player - position_initial) ** 2

        if priors:
            all_loss = loss + player_position_loss
        else:
            all_loss = loss

        loss_list.append(loss)

        return all_loss

    if spin:
        res = scipy.optimize.minimize(
            f_spin, initg, bounds=bounds,
            method='Powell')
    else:
        res = scipy.optimize.minimize(
            f_ours, initg, bounds=bounds,
            method='Powell')

    return res, loss_list


import scipy.optimize


def solve_trajectory_synthetic(shot_number, camera_params_train, synthetic, spin=False, priors=True):
    # find shot information
    game = synthetic.iloc[shot_number]["game"]
    clip = synthetic.iloc[shot_number]["clip"]

    #     start = ball_df.iloc[shot_number]["start"]
    #     end = ball_df.iloc[shot_number]["end"]

    homography, rotation_matrix, tvecs, cam_mtx, dist = get_important_cam_params(game, clip, camera_params_train)

    traj = np.array(eval(synthetic.iloc[i]["projection"]))
    start_position = np.array(eval(synthetic.iloc[0]["trajectory3D"]))[0, :]

    xi = traj[0, :]
    xf = traj[-1, :]

    loss_list = []

    x_scale = 10.97 / 2
    y_scale = 23.77 / 2

    fps = 30
    substeps = 10
    N = len(traj)

    if spin:
        bounds = [(-x_scale - 2, x_scale + 2), (-y_scale - 2, y_scale + 2), (0.0, 1.5), (-20, 20), (-40, 40), (-5, 10),
                  (-10 * 2 * math.pi, 10 * 2 * math.pi), (-1 * 2 * math.pi, 1 * 2 * math.pi),
                  (-5 * 2 * math.pi, 5 * 2 * math.pi)]
        initg = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    else:
        bounds = [(-x_scale - 2, x_scale + 2), (-y_scale - 2, y_scale + 2), (0.0, 1.5), (-20, 20), (-40, 40),
                  (-5, 10)]  # , (0, 1)
        initg = [1, 1, 1, 1, 1, 1]

    def f_spin(p):

        position_initial = np.array(p[:3])
        position = np.array(p[:3])
        vel = np.array(p[3:6])
        spin = np.array(p[6:9])

        D = 0.0012
        m = 0.056
        g = [0.0, 0.0, -9.82]
        magnus_coefficient = 0.00041

        N_max = N * 3
        t_max = N / fps
        dt = t_max / N_max

        positions = []

        pixel_err = 0
        tid = 0

        for t in range(N_max - 1):

            if t % 3 == 0:
                new_position = project_single_point_numpy(position, rotation_matrix, tvecs, cam_mtx, dist)
                positions.append(new_position)

                tid += 1
            v_norm = np.linalg.norm(vel)
            magnus_force = magnus_coefficient * np.cross(spin, vel)

            drag_force = - 1 * (D / m) * v_norm * vel
            a = g + drag_force + magnus_force #/ m
            vel = vel + a * dt
            position = position + vel * dt + 0.05 * a * dt ** 2

        loss = mean_squared_error(positions, traj)

        #         player1, player2 = get_players(pose_numpy, start)
        #         closest_player = find_closest_player(np.array(player1), np.array(player2), traj )

        #         homography_player = find_homography(closest_player, game, clip, camera_params_train)
        #         homography_player = np.array([homography_player[0], homography_player[1], 1.5])

        player_position_loss = np.linalg.norm(start_position - position_initial) ** 2

        if priors:
            all_loss = loss + player_position_loss
        else:
            all_loss = loss

        loss_list.append(loss)

        return all_loss

    def f_ours(p):

        position_initial = np.array(p[:3])
        position = np.array(p[:3])
        vel = np.array(p[3:6])

        D = 0.0012
        m = 0.056
        g = [0.0, 0.0, -9.82]

        N_max = N * 3
        t_max = N / fps
        dt = t_max / N_max

        positions = []

        pixel_err = 0
        tid = 0

        for t in range(N_max - 1):

            if t % 3 == 0:
                new_position = project_single_point_numpy(position, rotation_matrix, tvecs, cam_mtx, dist)
                positions.append(new_position)

                tid += 1
            v_norm = np.linalg.norm(vel)
            a = g - 1 * (D / m) * v_norm * vel
            vel = vel + a * dt
            position = position + vel * dt + 0.05 * a * dt ** 2

        loss = mean_squared_error(positions, traj)

        player_position_loss = np.linalg.norm(start_position - position_initial) ** 2

        if priors:
            all_loss = loss + player_position_loss
        else:
            all_loss = loss

        loss_list.append(loss)

        return all_loss

    if spin:
        res = scipy.optimize.minimize(
            f_spin, initg, bounds=bounds,
            method='Powell')
    else:
        res = scipy.optimize.minimize(
            f_ours, initg, bounds=bounds,
            method='Powell')

    return res, loss_list