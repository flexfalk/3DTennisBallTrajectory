import json
import ast
import matplotlib.pyplot as plt
import numpy as np
import ast
import math
import pandas as pd
import glob


def front_box_around_court(corners, width):
    corner1, corner2, corner3, corner4 = corners

    x1, y1 = corner1
    x2, y2 = corner2
    x3, y3 = corner3
    x4, y4 = corner4

    mid_back_x = (x1 + x2) / 2
    mid_back_y = (y1 + y2) / 2
    mid_back = (mid_back_x, mid_back_y)

    mid_front_x = (x3 + x4) / 2
    mid_front_y = (y3 + y4) / 2
    mid_front = (mid_front_x, mid_front_y)

    courtlength = distance_between_points(mid_back, mid_front)
    frontline = distance_between_points(corner3, corner4)
    backline = distance_between_points(corner1, corner2)

    sides_push = width
    backline_push = 0
    frontline_push = 0

    new_x1 = x1 + (-backline * sides_push)
    new_x2 = x2 + (backline * sides_push)
    new_x3 = x3 + (-frontline * sides_push)
    new_x4 = x4 + (frontline * sides_push)

    new_y1 = y1 + (-courtlength * backline_push)
    new_y2 = y2 + (-courtlength * backline_push)
    new_y3 = y3 + (courtlength * frontline_push)
    new_y4 = y4 + (courtlength * frontline_push)

    new_corners = [(new_x1, new_y1), (new_x2, new_y2), (new_x3, new_y3), (new_x4, new_y4)]
    return new_corners


def box_around_court(corners):
    corner1, corner2, corner3, corner4 = corners

    x1, y1 = corner1
    x2, y2 = corner2
    x3, y3 = corner3
    x4, y4 = corner4

    mid_back_x = (x1 + x2) / 2
    mid_back_y = (y1 + y2) / 2
    mid_back = (mid_back_x, mid_back_y)

    mid_front_x = (x3 + x4) / 2
    mid_front_y = (y3 + y4) / 2
    mid_front = (mid_front_x, mid_front_y)

    courtlength = distance_between_points(mid_back, mid_front)
    frontline = distance_between_points(corner3, corner4)
    backline = distance_between_points(corner1, corner2)

    sides_push = 0.30
    backline_push = 0
    frontline_push = 0

    new_x1 = x1 + (-backline * sides_push)
    new_x2 = x2 + (backline * sides_push)
    new_x3 = x3 + (-frontline * sides_push)
    new_x4 = x4 + (frontline * sides_push)

    new_y1 = y1 + (-courtlength * backline_push)
    new_y2 = y2 + (-courtlength * backline_push)
    new_y3 = y3 + (courtlength * frontline_push)
    new_y4 = y4 + (courtlength * frontline_push)

    new_corners = [(new_x1, new_y1), (new_x2, new_y2), (new_x3, new_y3), (new_x4, new_y4)]
    return new_corners


# Function to check if a point is inside the trapezoid
def is_inside_court(p, corners):
    x, y = p
    n = len(corners)
    inside = False

    p1x, p1y = corners[0]
    for i in range(n + 1):
        p2x, p2y = corners[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def midpoint(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    return (mid_x, mid_y)


def create_front_box(corners, threshold):
    corner1, corner2, corner3, corner4 = corners

    left_line = evenly_distributed_points(corner3, corner1, 100)

    right_line = evenly_distributed_points(corner4, corner2, 100)

    net_left = left_line[threshold]
    net_right = right_line[threshold]

    return [net_left, net_right, corner3, corner4]


# for now, lets just create it such that we say, if between feet are inside this box (that is not defined yet)

def filter_pose(all_poses, court_corners):
    players = []
    for pose in all_poses:

        keypoints = pose["keypoints"]
        right_foot = keypoints[15]
        left_foot = keypoints[16]

        middle = midpoint(right_foot, left_foot)
        #         print(middle)

        if not is_inside_court(middle, court_corners):
            #             print("not_inside")
            continue

        players.append(pose)

    return players


import math


def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def find_frontline(court_corners):
    corner1, corner2, corner3, corner4 = court_corners

    points = evenly_distributed_points(corner3, corner4, 10)

    return points


def find_backline(court_corners):
    corner1, corner2, corner3, corner4 = court_corners

    points = evenly_distributed_points(corner1, corner2, 10)

    return points


# def find_player1(all_poses, court_corners, game):

#     return all_poses[0]

def find_player1(all_poses, court_corners, game):
    # firt check if insidee
    best_pose = None
    minimum_distance = 10000
    for pose in all_poses:
        keypoints = pose["keypoints"]

        #         head = keypoints[0]
        right_foot = keypoints[15]
        left_foot = keypoints[16]

        middle = midpoint(right_foot, left_foot)

        front_corners = create_front_box(court_corners, 70)
        box_front_corners = front_box_around_court(front_corners, 0.15)

        # check if inside
        if is_inside_court(middle, box_front_corners):
            best_pose = pose
            return best_pose

        frontline = find_frontline(court_corners)
        #         important_points = find_important_points_front(game)

        #         points_to_check = frontline + important_points

        for point in frontline:

            distance = distance_between_points(middle, point)
            #             print(distance)
            if distance < minimum_distance:
                #                 print(distance)
                best_pose = pose
                minimum_distance = distance
    return best_pose


def find_player2(all_poses, court_corners, box_around_court, game):
    best_pose = None
    minimum_distance = 10000
    for pose in all_poses:

        keypoints = pose["keypoints"]

        right_foot = keypoints[15]
        left_foot = keypoints[16]

        middle = midpoint(right_foot, left_foot)

        if is_inside_court(middle, box_around_court):
            best_pose = pose
            return best_pose

        backline = find_backline(court_corners)

        for point in backline:

            distance = distance_between_points(middle, point)
            if distance < minimum_distance:
                best_pose = pose
                minimum_distance = distance

    return best_pose


def filter_pose_dist_from_lines(all_poses, court_corners, game):
    player1 = find_player1(all_poses, court_corners, game)

    all_poses.remove(player1)

    box_corners = box_around_court(court_corners)

    player2 = find_player2(all_poses, court_corners, box_corners, game)

    return (player1, player2)



def evenly_distributed_points(point1, point2, n):
    # Convert points to numpy arrays for easier manipulation
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Calculate the step size for interpolation
    step = 1.0 / (n + 1)

    # Initialize an empty list to store the evenly distributed points
    points = []

    # Interpolate points between p1 and p2
    for i in range(1, n + 1):
        new_point = p1 + (p2 - p1) * step * i
        points.append(new_point)

    return points


def find_important_points_front(game, clip):
    acourt = pd.read_csv(f"/kaggle/input/court-poles/Dataset/{game}/{clip}/court.csv")

    center_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[13])

    left_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[10])
    right_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[11])

    return [center_front, left_front, right_front]


def find_important_points_back(game, clip):
    acourt = pd.read_csv(f"/kaggle/input/court-poles/Dataset/{game}/{clip}/court.csv")

    center_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[13])

    left_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[10])
    right_front = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[11])

    return [center_front, left_front, right_front]


def find_court_corners(game, clip):
    #     game = "game4"
    acourt = pd.read_csv(f"/kaggle/input/court-poles/Dataset/{game}/{clip}/court.csv")

    # plt.imshow(img)
    corner1 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[0])
    corner2 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[1])
    corner3 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[2])
    corner4 = tuple(acourt[["x-coordinate", "y-coordinate"]].loc[3])

    corners = [corner1, corner2, corner3, corner4]

    return corners

def create_pose_csv(clip_path):
    frames = glob.glob(clip_path + '/*')
    frames.sort()
    #     print(frames)
    filtered_poses_video = []
    for frame in frames:
        game = clip_path.split("/")[-2]
        clip = clip_path.split("/")[-1]

        #         print(frame)
        with open(frame, 'r') as file:
            poses = [ast.literal_eval(element) for element in file.read().split("\n")[:-1]]

        court_corners = find_court_corners(game, clip)

        filtered_poses = filter_pose_dist_from_lines(poses, court_corners, game)
        #print(filtered_poses)
        player1 = filtered_poses[0]["keypoints"]
        player2 = filtered_poses[1]["keypoints"]

        player1 = [item for sublist in player1 for item in sublist]
        player2 = [item for sublist in player2 for item in sublist]

        players = player1 + player2

        filtered_poses_video.append(players)

    return filtered_poses_video