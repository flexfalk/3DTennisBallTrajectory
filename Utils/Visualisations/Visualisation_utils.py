import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Utils.FilterPoses.FilterPoses_utils import find_court_corners
import torch
from Utils.Reconstruction3D.Reconstruction3D_utils import get_court_dimension

def inference_on_clip(clip_path, preds=None, ball=True, pose=True, hits=True, whitescreen=True, corners=True,
                      darkmode=True):
    replacements = {"tennis-tracknet-videos": "filtered-poses", "video.mp4": "data.csv"}
    game = clip_path.split("/")[5]
    clip = clip_path.split("/")[6]

    # change later
    save_path = "lol.mp4"

    hits_path = clip_path

    for old, new in replacements.items():
        hits_path = hits_path.replace(old, new)

    cap = cv2.VideoCapture(clip_path)

    n_frame = 0

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (width, height)

    result = cv2.VideoWriter(save_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             20, size)

    img_list = []

    data = pd.read_csv(hits_path)
    _hits = data["hits"]
    if preds:
        _hits = preds

    _ball = data[["ball_x", "ball_y"]]
    #     print(data.columns)
    _pose = data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                  '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                  '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                  '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
                  '59', '60', '61', '62', '63', '64', '65', '66', '67']]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # cv.waitKey(100)
        try:
            element = _hits[n_frame]
        except IndexError:
            print("Cant recieve element in preds")
            cap.release()
            # result.release()
        #             cv2.destroyAllWindows()
        if whitescreen:
            if darkmode:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 1
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        if hits:
            # color screen on hit
            if element == 1:
                frame[:, :, 0] = 255  # frame turn blue
            if element == 2:
                frame[:, :, 2] = 255  # frame turns red

        # paint ball
        if ball:
            current_ball = _ball.iloc[n_frame].tolist()
            cv2.circle(frame, (current_ball[0], current_ball[1]), 10, (0, 255, 0), -1)  # Green circle

        if pose:
            # paint pose
            current_pose = np.array(_pose.iloc[n_frame].tolist()).reshape((34, 2))

            # Draw circles for keypoints
            for i, keypoint in enumerate(current_pose):
                if i > 16:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, color, -1)

            # Draw lines to connect keypoints
            connections = [[0, 1], [0, 2], [2, 1], [1, 3], [2, 4], [4, 6], [6, 8], [8, 10],
                           [6, 5], [6, 12], [12, 11], [12, 14], [14, 16], [15, 13], [13, 11],
                           [11, 5], [5, 7], [7, 9], [5, 3]]

            connections2 = [(x + 17, y + 17) for (x, y) in connections]

            for connection in connections:
                cv2.line(frame, (int(current_pose[connection[0]][0]), int(current_pose[connection[0]][1])),
                         (int(current_pose[connection[1]][0]), int(current_pose[connection[1]][1])), (128, 128, 128), 1)

            for connection in connections2:
                cv2.line(frame, (int(current_pose[connection[0]][0]), int(current_pose[connection[0]][1])),
                         (int(current_pose[connection[1]][0]), int(current_pose[connection[1]][1])), (128, 128, 128), 1)

        if corners:
            _corners = find_court_corners(game, clip)
            connections3 = [[_corners[0], _corners[1]], [_corners[0], _corners[2]], [_corners[1], _corners[3]],
                            [_corners[2], _corners[3]]]
            # paint court
            for corner in _corners:
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (102, 102, 153), -1)

            for connection in connections3:
                cv2.line(frame, (int(connection[0][0]), int(connection[0][1])),
                         (int(connection[1][0]), int(connection[1][1])), (128, 128, 128), 1)
        n_frame += 1

        result.write(frame)

    result.release()

def plot_tennis_court(ax):


    # Tennis court dimensions

    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    # Baseline
    ax.plot([-half_court_width, half_court_width], [-half_court_length, -half_court_length], [0, 0], color='black')
    # Sidelines
    ax.plot([-half_court_width, -half_court_width], [-half_court_length, half_court_length], [0, 0], color='black')
    ax.plot([half_court_width, half_court_width], [-half_court_length, half_court_length], [0, 0], color='black')
    #     # Service lines
    ax.plot([-half_court_width, half_court_width], [0, 0], [0, 0], color='black')
    #     # Center service line
    # #     ax.plot([court_length / 2, court_length / 2], [0, court_width], [0, 0], color='black')
    #     # Backline
    ax.plot([-half_court_width, half_court_width], [half_court_length, half_court_length], [0, 0], color='black')

def shot_plotter(image_path: str, projected_path: torch.tensor, trajectory_3d: torch.tensor, labels : torch.tensor, with_label = True):

    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    img = cv2.imread(image_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img)
    for i in range(len(projected_path)):
        s = (i + 1) * 2
        #             print(s)
        axs[0].scatter(projected_path[i, 0], projected_path[i, 1], s=s, color="green", label="Reprojected Shot")
        if with_label:
            axs[0].scatter(labels[i, 0], labels[i, 1], s=s, color="red", label="True Shot")
        axs[0].set_title("Reprojection on real shot")

    # Plot the trajectory
    axs[1] = fig.add_subplot(122, projection='3d')
    for i in range(len(trajectory_3d)):
        s = (i + 1) * 2  # Increase size with each iteration
        color = 'green' if trajectory_3d[i, 2] > 0 else 'black'
        axs[1].scatter(trajectory_3d[i, 0], trajectory_3d[i, 1], trajectory_3d[i, 2], s=s, c=color)

    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_zlabel('Z')
    axs[1].set_title('Tennis Shot Trajectory with Scatter Points')
    axs[1].legend()
    plot_tennis_court(axs[1])

    court_length = 23.77  # meters
    court_width = 10.97  # meters

    axs[1].set_xlim(-half_court_length - 2, half_court_length + 2)  # Set x-axis limits
    axs[1].set_ylim(-half_court_length - 2, half_court_length + 2)  # Set y-axis limits
    axs[1].set_zlim(-1, 4)  # Set z-axis limits

    #         axs[1].view_init(elev=1, azim=1)  # Change the elevation (up-down) and azimuth (left-right) angles

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

    angles = [(0, 0), (90, -90)]
    for k in range(2):
        #             print(k)
        angle = angles[k]

        # Plot the trajectory
        axs2[k] = fig2.add_subplot(1, 2, k + 1, projection='3d')
        for i in range(len(trajectory_3d)):
            #                 print(s)
            s = (i + 1) * 2  # Increase size with each iteration
            color = 'green' if trajectory_3d[i, 2] > 0 else 'black'
            axs2[k].scatter(trajectory_3d[i, 0], trajectory_3d[i, 1], trajectory_3d[i, 2], s=s, c=color)

        axs2[k].set_xlabel('X')
        axs2[k].set_ylabel('Y')
        axs2[k].set_zlabel('Z')
        axs2[k].set_title('Tennis Shot Trajectory with Scatter Points')
        axs2[k].legend()
        plot_tennis_court(axs2[k])

        court_length = 23.77  # meters
        court_width = 10.97  # meters

        axs2[k].set_xlim(-half_court_length - 2, half_court_length + 2)  # Set x-axis limits
        axs2[k].set_ylim(-half_court_length - 2, half_court_length + 2)  # Set y-axis limits
        axs2[k].set_zlim(-1, 4)  # Set z-axis limits

        axs2[k].view_init(elev=angle[0],
                          azim=angle[1])  # Change the elevation (up-down) and azimuth (left-right) angles
    plt.tight_layout()
    plt.show()


def shot_plotter_for_report(image_path, projected_path, trajectory_3d, labels, with_label):
    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    img = cv2.imread(image_path)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the image with projected path and labels
    axs[0].imshow(img)
    for i in range(len(projected_path)):
        s = (i + 1) * 2
        axs[0].scatter(projected_path[i, 0], projected_path[i, 1], s=s, color="green",
                       label="Reprojected Shot" if i == 0 else "")
        if with_label:
            axs[0].scatter(labels[i, 0], labels[i, 1], s=s, color="red", label="True Shot" if i == 0 else "")
    axs[0].set_title("Reprojection on real shot")
    axs[0].legend()

    # Plot the first 3D trajectory
    ax1 = fig.add_subplot(132, projection='3d')
    for i in range(len(trajectory_3d)):
        s = (i + 1) * 2
        color = 'green' if trajectory_3d[i, 2] > 0 else 'black'
        ax1.scatter(trajectory_3d[i, 0], trajectory_3d[i, 1], trajectory_3d[i, 2], s=s, c=color)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Tennis Shot Trajectory 1')
    ax1.set_xlim(-half_court_length - 2, half_court_length + 2)
    ax1.set_ylim(-half_court_length - 2, half_court_length + 2)
    ax1.set_zlim(-1, 4)
    ax1.view_init(elev=0, azim=0)
    plot_tennis_court(ax1)

    # Plot the second 3D trajectory
    ax2 = fig.add_subplot(133, projection='3d')
    for i in range(len(trajectory_3d)):
        s = (i + 1) * 2
        color = 'green' if trajectory_3d[i, 2] > 0 else 'black'
        ax2.scatter(trajectory_3d[i, 0], trajectory_3d[i, 1], trajectory_3d[i, 2], s=s, c=color)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Tennis Shot Trajectory 2')
    ax2.set_xlim(-half_court_length - 2, half_court_length + 2)
    ax2.set_ylim(-half_court_length - 2, half_court_length + 2)
    ax2.set_zlim(-1, 4)
    ax2.view_init(elev=90, azim=-90)
    plot_tennis_court(ax2)

    plt.tight_layout()
    plt.show()

    return fig


def new_pose_plotter(image_path, filtered_poses_path):
    data = pd.read_csv(filtered_poses_path)

    #     print(data.columns)
    _pose = data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                  '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                  '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                  '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
                  '59', '60', '61', '62', '63', '64', '65', '66', '67']]

    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # paint pose
    current_pose = np.array(_pose.iloc[0].tolist()).reshape((34, 2))

    # Draw circles for keypoints
    for i, keypoint in enumerate(current_pose):
        if i > 16:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 9, color, -1)

    # Draw lines to connect keypoints
    connections = [[0, 1], [0, 2], [2, 1], [1, 3], [2, 4], [4, 6], [6, 8], [8, 10],
                   [6, 5], [6, 12], [12, 11], [12, 14], [14, 16], [15, 13], [13, 11],
                   [11, 5], [5, 7], [7, 9], [5, 3]]

    connections2 = [(x + 17, y + 17) for (x, y) in connections]

    for connection in connections:
        cv2.line(frame, (int(current_pose[connection[0]][0]), int(current_pose[connection[0]][1])),
                 (int(current_pose[connection[1]][0]), int(current_pose[connection[1]][1])), (128, 128, 128), 1)

    for connection in connections2:
        cv2.line(frame, (int(current_pose[connection[0]][0]), int(current_pose[connection[0]][1])),
                 (int(current_pose[connection[1]][0]), int(current_pose[connection[1]][1])), (128, 128, 128), 1)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.imshow(frame, cmap=None)

    return fig


def new_shot_plotter(image_path: str, labels: torch.tensor):
    court_length, court_width, half_court_length, half_court_width, net_height_middle, net_height_sides = get_court_dimension()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.imshow(img, cmap=None)
    for i in range(len(labels)):
        s = (len(labels) - i - 1) * 20
        alpha = 1 - (i / len(labels))
        #             print(s)
        axs.scatter(labels[i, 0], labels[i, 1], s=s, color="red", alpha=alpha)

    plt.show()
    return fig
