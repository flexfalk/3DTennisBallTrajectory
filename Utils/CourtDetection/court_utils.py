import os
import sys
import cv2
import numpy as np
from scipy.spatial import distance
import sympy
from sympy import Line
import os
import torch
import pandas as pd
import torch.nn.functional as F
import config_path



def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point


def is_point_in_image(x, y, input_width=1280, input_height=720):
    res = False
    if x and y:
        res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res


def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred


def refine_kps(img, x_ct, y_ct, crop_size=40):
    refined_x_ct, refined_y_ct = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    x_min = max(x_ct - crop_size, 0)
    x_max = min(img_height, x_ct + crop_size)
    y_min = max(y_ct - crop_size, 0)
    y_max = min(img_width, y_ct + crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)

    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if new_x_ct > 0 and new_x_ct < img_crop.shape[0] and new_y_ct > 0 and new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct
    return refined_y_ct, refined_x_ct


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [int((x1 + x3) / 2), int((y1 + y3) / 2), int((x2 + x4) / 2), int((y2 + y4) / 2)],
                            dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines



def make_court_conf():
    court_ref = CourtReference()
    refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

    court_conf_ind = {}
    for i in range(len(court_ref.court_conf)):
        conf = court_ref.court_conf[i + 1]
        inds = []
        for j in range(4):
            inds.append(court_ref.key_points.index(conf[j]))
        court_conf_ind[i + 1] = inds
    return court_conf_ind, court_ref, refer_kps

from court_reference import CourtReference


def get_trans_matrix(points):
    matrix_trans = None
    dist_max = np.Inf
    court_conf_ind, court_ref, refer_kps = make_court_conf()
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]

        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if not any([None in x for x in inters]):
            matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
            dists = []
            for i in range(12):

                if i not in inds and points[i][0] is not None:
                    dists.append(distance.euclidean(points[i], trans_kps[i].flatten()))
            dist_median = np.mean(dists)
            if dist_median < dist_max:
                matrix_trans = matrix
                dist_max = dist_median
    return matrix_trans, refer_kps

def run_on_image(game_path, model, use_refine_kps, use_homography):
    x_coords = []
    y_coords = []
    point_nr = []

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    input_path = os.path.join(game_path, '0000.jpg')

    image = cv2.imread(input_path)
    image_copy = image.copy()
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if use_homography:
        matrix_trans, refer_kps = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

    for j in range(len(points)):
        if points[j][0] is not None:
            x = int(points[j][0])
            y = int(points[j][1])

            x_coords.append(x)
            y_coords.append(y)
            point_nr.append(j)

            image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                               radius=0, color=(0, 0, 255), thickness=10)

    coords_dict = {
        'point': point_nr,
        'x-coordinate': x_coords,
        'y-coordinate': y_coords
    }

    df = pd.DataFrame(coords_dict)
    return df, image