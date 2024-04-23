import cv2
import numpy as np
import glob
import pandas as pd
import ast




def find_court_corners(game: str, clip: str):
    #     game = "game4"
    acourt = pd.read_csv(f"/kaggle/input/court-poles/Dataset/{game}/{clip}/court.csv")

    corner1 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[0])
    corner2 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[1])
    corner3 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[2])
    corner4 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[3])

    corners = [corner1, corner2, corner3, corner4]

    return corners


def find_poles_and_corners(game, clip):
    # find poles
    poles_and_corners = []
    with open(f"/kaggle/input/court-poles/Dataset/{game}/{clip}/poles.txt", "r") as file:
        for line in file:
            poles_and_corners.append([int(float(el)) for el in line.strip("\n").split(", ")])

    corners = find_court_corners(game, clip)

    foo = [[v[0], v[1]] for v in corners]
    poles_and_corners += foo

    return poles_and_corners


def find_poles_and_corners_world():
    # All these measurements are in meters
    court_length = 23.77
    court_width = 10.97
    half_court_length = court_length / 2
    half_court_width = court_width / 2
    net_height_middle = 0.91
    net_height_sides = 1.067

    # Find corners
    # left_bottom_corner = [0, 0, 0]
    # right_bottom_corner = [court_width, 0, 0]
    left_bottom_corner = [-half_court_width, -half_court_length, 0]
    right_bottom_corner = [half_court_width, -half_court_length, 0]

    # left_top_corner = [0, court_length, 0]
    # right_top_corner = [court_width, court_length, 0]
    left_top_corner = [-half_court_width, half_court_length, 0]
    right_top_corner = [half_court_width, half_court_length, 0]

    # Find poles
    # left_pole_top = [0 + 0.3, half_court_length, net_height_sides]
    # left_pole_bottom = [0 + 0.3, half_court_length, 0]
    left_pole_top = [-half_court_width + 0.3, 0, net_height_sides]
    left_pole_bottom = [-half_court_width + 0.3, 0, 0]

    # middle_net_top = [half_width_length, half_court_length, net_height_middle]
    # middle_net_bottom = [half_width_length, half_court_length, 0]
    middle_net_top = [0, 0, net_height_middle]
    middle_net_bottom = [0, 0, 0]

    # right_pole_top = [court_width - 0.3, half_court_length, net_height_sides]
    # right_pole_bottom = [court_width - 0.3, half_court_length, 0]
    right_pole_top = [half_court_width - 0.3, 0, net_height_sides]
    right_pole_bottom = [half_court_width - 0.3, 0, 0]

    return [left_pole_top, left_pole_bottom,
            middle_net_top, middle_net_bottom,
            right_pole_top, right_pole_bottom,
            left_top_corner, right_top_corner,
            left_bottom_corner, right_bottom_corner]


def find_camera_matrix(game: str, clip: str):
    image_points = find_poles_and_corners(game, clip)
    image_points = [np.array(image_points, dtype=np.float32)]

    world_points = find_poles_and_corners_world()
    world_points = [np.array(world_points, dtype=np.float32)]

    width = 1280
    height = 720

    # Approximate focal length based on image size and field of view
    fov_degrees = 60  # Assuming a 60-degree field of view
    focal_length_pixels = (width / 2) / np.tan(np.deg2rad(fov_degrees / 2))

    # Initialize camera matrix guess with approximate focal length and principal point at image center
    camera_matrix_guess = np.array([[focal_length_pixels, 0, width / 2],
                                    [0, focal_length_pixels, height / 2],
                                    [0, 0, 1]], dtype=np.float32)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, (1280, 720), camera_matrix_guess,
                                                       None, flags=flags)

    ground_points = [1,3,5,6,7,8,9]
    image_points_homography = np.array([image_points[0][i] for i in ground_points])
    world_points_homography = np.array([world_points[0][i] for i in ground_points])
    homography, _ = cv2.findHomography(image_points_homography, world_points_homography)

    return ret, mtx.tolist(), dist.tolist(), rvecs[0].tolist(), tvecs[0].tolist(), homography.tolist()
