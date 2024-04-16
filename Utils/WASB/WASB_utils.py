
import cv2
import numpy as np

def getInputArr( path ,path1 ,path2 , width , height):
    try:
        #read the image
        img = cv2.imread(path, 1)
        #resize it
        img = cv2.resize(img, ( width , height ))
        #input must be float type
        img = img.astype(np.float32)
        #read the image
        img1 = cv2.imread(path1, 1)
        #resize it
        img1 = cv2.resize(img1, ( width , height ))
        #input must be float type
        img1 = img1.astype(np.float32)
        #read the image
        img2 = cv2.imread(path2, 1)
        #resize it
        img2 = cv2.resize(img2, ( width , height ))
        #input must be float type
        img2 = img2.astype(np.float32)
        #combine three imgs to  (width , height, rgb*3)
        imgs =  np.concatenate((img, img1, img2),axis=2)
        #since the odering of TrackNet  is 'channels_first', so we need to change the axis
        imgs = np.rollaxis(imgs, 2, 0)
        imgs = np.expand_dims(imgs, axis=0)
        #print(imgs.shape)
        return imgs
    except Exception as e:
        print(path , e)
