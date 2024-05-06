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


def create_3d_trajectory(output, N):
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

    for i in range(N_max - 1):
        v_norm = torch.norm(v)

        a = g - 1 * (D / m) * v_norm * v
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