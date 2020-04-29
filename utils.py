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
        cv2.imread(path1, cv2.IMREAD_GRAYSCALE),
        cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    )

def detect_edges(image_tuple):
    return (
        cv2.Laplacian(image_tuple[0], -1),
        cv2.Laplacian(image_tuple[1], -1)
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

def get_time_slices(framelist, fps, threshold):
    #   Returns a list of TimeSlices.
    #   'framelist' is a sorted list of anormal frames.
    #   'fps' is how many fps were sampled from the original video
    #   'threshold' determins how many empty frames before cutting the clip

    framelist.reverse() #reverse because we are pop-ing, and want to start from the front

    min_delta = threshold
    frametime = 1000/fps
    time_slices = []

    legal_delta = min_delta
    start_frame = framelist.pop()

    while len(framelist) > 0:
        next_frame = framelist.pop()
        delta_frame = next_frame-start_frame

        if delta_frame > legal_delta or len(framelist) == 0:
            time_slices.append(
                TimeSlice(
                    #Adding 'threshold' frames padding either side of the slice
                    (start_frame - threshold) * frametime,
                    (start_frame + threshold) + legal_delta * frametime
                )
            )
            #Reset values for next slice
            start_frame = next_frame
            legal_delta = min_delta
        else:
            legal_delta = min_delta + delta_frame

    return time_slices

def filter_frames(frames, threshold):
    #Removes frames with no neighbours within the given threshold.

    new_frames = []

    for idx in range(1, len(frames)-1):
        space = frames[idx+1] - frames[idx-1]

        if space <= threshold :
            new_frames.append(frames[idx])
    
    return new_frames

#
#   FRAME DIFFERENCE FUNCTIONS
#

def mean_squared_error(path1, path2):
    (image1, image2) = detect_edges(load_images(path1, path2))

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image1.shape[1])

def psnr(path1, path2):
    mse = mean_squared_error(path1, path2)

    if mse == 0:
        return (20*math.log10(255))
    else:
        return (20*math.log10(255) - 10*np.log10(mse))

def ssim(path1, path2):
    (image1, image2) = detect_edges(load_images(path1, path2))
    (score, _) = compare_ssim(image1, image2, full=True)

    #Convert to dissimilarity since we are more interested if there is a difference
    #rather than then images being similar.
    return (1-score) / 2

def optical_flow_field(path1, path2):
    (image1, image2) = load_images(path1, path2)

    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    avg_magnitude = np.average(magnitude)
    avg_angle = np.average(angle)

    return (avg_magnitude, avg_angle)
