import torch
import yaml
import config_path
from attrdict import AttrDict
import cv2
import numpy as np
import glob
import time
import os
from Utils.WASB.WASB_utils import getInputArr
from Utils.WASB.hrnet import HRNet
import pandas as pd




model_path = 'Utils/WASB/wasb_tennis_best.pth.tar'

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

device = 'cpu'

config_path = 'Utils/WASB/wasb.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
cfg = AttrDict(cfg)
model = HRNet(cfg)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Path to Input Data.
# Place data in the data folder
images_path = 'Data/dataset'


# Path for output data
out_dataset_path = 'Data/output'

# Output if you want a video as well
output_video_path = '/Users/Morten/School/Masters/Thesis/WASB/lol3.mp4'

output_width = 512
output_height = 288

input_width = 1280
input_height = 720

width_scale = input_width / output_width
height_scale = input_height / output_height

# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (input_width, input_height))

if not os.path.isdir(out_dataset_path):
    os.makedirs(out_dataset_path)


games = glob.glob(images_path + '/*')
print(games)

heatmaps = []
# print(dirs)
for game in games:

    start = time.time()
    if game.endswith('.docx'):
        continue

    # create game directory in output
    game_number = game.split('/')[-1]
    game_path = os.path.join(out_dataset_path, game_number)

    if not os.path.isdir(game_path):
        os.mkdir(game_path)

    clips = glob.glob(game + '/*')

    for clip in clips:

        clip_number = clip.split('/')[-1]
        clip_path = os.path.join(game_path, clip_number)

        if not os.path.isdir(clip_path):
            os.mkdir(clip_path)

        coordiantes_path = os.path.join(clip_path, 'Label.csv')

        header = ['file name', 'x-coordinate', 'y-coordinate']

        coordinates = []

        # get all JPG images in the path
        images = glob.glob(clip + "/*.jpg")
        images.sort()

        one_img_left = False
        two_img_left = False


        # predict each images
        # since wasb takes 3 images as input, we start at
        #for i in range(2, len(images), 3):
        for i in range(0, len(images), 3):



            if i + 1 == len(images):
                one_img_left = True
                first_idx = i
                second_idx = i - 1
                third_idx = i - 2

            elif i + 2 == len(images):
                two_img_left = True
                first_idx = i
                second_idx = 2
                third_idx = i-1

            else:
                first_idx = i
                second_idx = i+1
                third_idx = i+2

            # print(f'I {i}')
            # print(f'len images {len(images)}')
            #
            #
            # print(f'first index {first_idx}')
            # print(f'second index {second_idx}')
            # print(f'third index {third_idx}')
            # print('----------------------------------------')

            filename_1 = images[first_idx].split('/')[-1]
            filename_2 = images[second_idx].split('/')[-1]
            filename_3 = images[third_idx].split('/')[-1]






            # name of the image
            filename_1 = images[first_idx].split('/')[-1]
            filename_2 = images[second_idx].split('/')[-1]
            filename_3 = images[third_idx].split('/')[-1]

            # load input data
            output_img_1 = cv2.imread(images[first_idx])
            output_img_2 = cv2.imread(images[second_idx])
            output_img_3 = cv2.imread(images[third_idx])

            X = getInputArr(images[first_idx], images[second_idx], images[third_idx], output_width, output_height)


            print('before pred')
            # We get three predictions
            out = model(torch.from_numpy(X).float().to(device))[0]
            print('after pred')


            # start with just predictions from first frame:
            out_3 = out[0,2,:,:].detach().cpu().numpy()
            out_2 = out[0, 1, :, :].detach().cpu().numpy()
            out_1 = out[0, 0, :, :].detach().cpu().numpy()



            #out_3 = out[0,0,:,:].detach().cpu().numpy()
            #out_2 = out[0, 1, :, :].detach().cpu().numpy()
            #out_1 = out[0, 2, :, :].detach().cpu().numpy()


            # Calculate max values

            max_3 = np.max(out_3)
            max_2 = np.max(out_2)
            max_1 = np.max(out_1)




            # Find the indices of the maximum value
            if max_1 <= -1:
                y_1, x_1 = 0, 0
            else:
                max_index_1 = np.argmax(out_1)
                max_index_1_2d = np.unravel_index(max_index_1, out_1.shape)
                y_1 = int(max_index_1_2d[0] * height_scale)
                x_1 = int(max_index_1_2d[1] * width_scale)

            if max_2 <= -1:
                y_2, x_2 = 0, 0
            else:
                max_index_2 = np.argmax(out_2)
                max_index_2_2d = np.unravel_index(max_index_2, out_2.shape)
                y_2 = int(max_index_2_2d[0] * height_scale)
                x_2 = int(max_index_2_2d[1] * width_scale)

            if max_3 <= -1:
                y_3, x_3 = 0, 0
            else:

                max_index_3 = np.argmax(out_3)
                max_index_3_2d = np.unravel_index(max_index_3, out_3.shape)
                y_3 = int(max_index_3_2d[0] * height_scale)
                x_3 = int(max_index_3_2d[1] * width_scale)


            if one_img_left:
                coordinates.append([filename_1, x_1, y_1])
                # cv2.circle(output_img_1, (x_1, y_1), 10, [255, 0, 0], -1)
                # output_video.write(output_img_1)
            elif two_img_left:
                coordinates.append([filename_1, x_1, y_1])
                coordinates.append([filename_2, x_2, y_2])
                # cv2.circle(output_img_1, (x_1, y_1), 10, [255, 0, 0], -1)
                # cv2.circle(output_img_2, (x_2, y_2), 10, [255, 0, 0], -1)
                # output_video.write(output_img_1)
                # output_video.write(output_img_2)

            else:
                coordinates.append([filename_1, x_1, y_1])
                coordinates.append([filename_2, x_2, y_2])
                coordinates.append([filename_3, x_3, y_3])

                # cv2.circle(output_img_1, (x_1,y_1), 10, [255,0,0], -1)
                # cv2.circle(output_img_2, (x_2, y_2), 10, [255, 0, 0], -1)
                # cv2.circle(output_img_3, (x_3, y_3), 10, [255, 0, 0], -1)
                #
                # output_video.write(output_img_1)
                # output_video.write(output_img_2)
                # output_video.write(output_img_3)

        df = pd.DataFrame(coordinates, columns=header)
        if len(images) != len(df):
            print('MISTAKE')
        df.to_csv(coordiantes_path)

# output_video.release()