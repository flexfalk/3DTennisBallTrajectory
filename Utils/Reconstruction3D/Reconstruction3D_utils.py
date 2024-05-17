import pandas as pd
import ast
import torch
import numpy as np

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


def create_3d_trajectory(output, N, until_ground_hit=False):
    position = output[:, 0:3]
    v = output[:, 3:6]

    D = 0.00114
    #     D = output[:, -1]
    m = 0.056
    t = 0  # Start time
    N_max = 3 * N  # How many points we want per frame
    t_max = N / 25
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


def create_3d_trajectory_with_spin(output, N, until_ground_hit=False):
    position = output[:, 0:3]
    v = output[:, 3:6]

    spin = output[:, 6:9]

    D = 0.00114
    #     D = output[:, -1]
    m = 0.056
    t = 0  # Start time
    N_max = 3 * N  # How many points we want per frame
    t_max = N / 25
    delta_t = t_max / N_max  # Time interval

    g = torch.tensor([0.0, 0.0, -9.81], device=output.device)

    magnus_coefficient = 0.0004

    # positions = torch.tensor(position, device=output.device).view(1, 3)
    # positions = torch.tensor(position, device=output.device).clone().view(1, 3)
    positions = position.clone().view(1, 3)

    if until_ground_hit:
        while position[0, 2] > 0:
            v_norm = torch.norm(v)

            magnus_force = magnus_coefficient * torch.cross(spin, v)

            drag_force = - 1 * (D / m) * v_norm * v

            a = g + drag_force + magnus_force / m

            v = v + a * delta_t
            position = position + v * delta_t + 0.5 * a * delta_t ** 2
            positions = torch.cat((positions, position.view(1, 3)), dim=0)

    else:
        for i in range(N_max - 1):
            v_norm = torch.norm(v)

            magnus_force = magnus_coefficient * torch.cross(spin, v)

            drag_force = - 1 * (D / m) * v_norm * v

            a = g + drag_force + magnus_force / m

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
        vy = np.random.uniform(10, 40)
        vz = np.random.uniform(0, 5)

        init_params = [x, y, z, vx, vy, vz]
        front_player_shots.append(init_params)

    for i in range(N_back_player_shots):
        x = np.random.uniform(-half_court_width - 1, half_court_width + 1)
        y = np.random.uniform(1, half_court_length + 1)
        z = np.random.uniform(0.1, 3)

        vx = np.random.uniform(-4, 4)
        vy = np.random.uniform(-10, -40)
        vz = np.random.uniform(0, 5)

        init_params = [x, y, z, vx, vy, vz]
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

import cv2
def ball_hits_court(pred_traj, true_traj, homography_matrix):
    """
    This function dvivides the court in 12 square, three in width and 4 in height, meaning each half of the court consists of 6 squares.
    The squares are numbered from top left going right.

    pred_traj : the predicted 3D trajectory
    true_traj : the true 2D trajectory
    homography_matrix : the homography matrix for the given shot
    return : a number determining which square of the field the true and predicted ball lands in
    """

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