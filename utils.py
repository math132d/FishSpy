import math

from os import path, listdir
from skimage.measure import compare_ssim

import cv2
import numpy as np

from TimeSlice import TimeSlice

#
#   FILE SYSTEM UTILS
#

def load_images(path1, path2):
    return (
        cv2.equalizeHist(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)),
        cv2.equalizeHist(cv2.imread(path2, cv2.IMREAD_GRAYSCALE))
    )

def files_from(folder):
    #Returns sorted list of files in the defined folder
    return sorted(list(filter(
        lambda file: path.isfile(path.join(folder, file)),
        listdir(folder)
    )))

#
#   TIME RELATED FUNCTIONS
#

def get_time_slices(framelist, fps):
    #   Returns a list of TimeSlices.
    #   'framelist' is a sorted list of anormal frames.
    #   'fps' is how many fps were sampled from the original video

    framerate = 25/fps
    time_slices = []
    last_frame = -1

    for this_frame in framelist:
        if last_frame < 0:
            last_frame = this_frame

        if this_frame-last_frame > (framerate * 3):
            time_slices.append(TimeSlice(last_frame * framerate, this_frame * framerate))
            last_frame = -1 #Make sure the next frame gets assigned to 'last_frame'

    return time_slices

#
#   FRAME DIFFERENCE FUNCTIONS
#

def mean_squared_error(path1, path2):
    (image1, image2) = load_images(path1, path2)

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image1.shape[1])

def psnr(path1, path2):
    mse = mean_squared_error(path1, path2)
    return 20*math.log10(255) - np.log10(mse)

def ssim(path1, path2):
    (image1, image2) = load_images(path1, path2)
    (score, _) = compare_ssim(image1, image2, full=True)

    cv2.imshow("asd", image1)

    #Inverse because SSIM is usually made for evaluating compression artifacts,
    #meaning differences are bad
    return 1.0/score
