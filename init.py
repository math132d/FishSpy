from os import path, listdir

import cv2
import numpy as np

def mean_squared_error(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image2.shape[1])

FRAMES_PATH = path.abspath("./vid/frames/")

FRAMES_LIST = list(filter( #Include only entries that are files, not folders
    lambda file: path.isfile(path.join(FRAMES_PATH, file)),
    listdir(FRAMES_PATH)
))

print("Looking for frames in: " + FRAMES_PATH)

for idx in range(len(FRAMES_LIST)-1):
    mean_error = mean_squared_error(
        path.join(FRAMES_PATH, FRAMES_LIST[idx]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx+1])
    )
