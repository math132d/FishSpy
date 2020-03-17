import cv2
import numpy as np

from os import path, listdir


def mean_squared_error(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image2.shape[1])

frames_path = path.abspath("./vid/frames/")

frames_list = list( filter( #Include only entries that are files, not folders
    lambda file: path.isfile(path.join(frames_path, file)), 
    listdir(frames_path)
))

print("Looking for frames in: " + frames_path)

for idx in range(len(frames_list)-1):
    mean_error = mean_squared_error(
            path.join(frames_path, frames_list[idx]),
            path.join(frames_path, frames_list[idx+1])
        )
