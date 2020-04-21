from os import path, listdir

import cv2
import numpy as np

def files_from(folder):
    #Returns sorted list of files in the defined folder
    return list(filter(
        lambda file: path.isfile(path.join(folder, file)),
        listdir(folder)
    )).sort()

def mean_squared_error(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image1.shape[1])
