import cv2
import numpy as np
import pandas as pd
from Utils.FilterPoses.FilterPoses_utils import find_court_corners

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