from os import path
import matplotlib.pyplot as plt

from TimeSlice import TimeSlice;

import cv2
import numpy as np

import utils


first = TimeSlice(0, 4000)
second = TimeSlice(2000, 2000)

print(first.intersection_over_union(second));

# FRAMES_PATH = path.abspath("./vid/frames/")
# FRAMES_LIST = utils.files_from(FRAMES_PATH)

# magnitudes = []
# angles = []

# for idx in range(len(FRAMES_LIST)-1):
#     (prev_img, next_img) = utils.load_images(
#         path.join(FRAMES_PATH, FRAMES_LIST[idx]),
#         path.join(FRAMES_PATH, FRAMES_LIST[idx+1])
#     )

#     flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 5, 3, 5, 1.2, 0)
#     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#     magnitudes.append(np.average(magnitude))
#     angles.append(np.average(angle))

    # mask = np.zeros((
    #     prev_img.shape[0],
    #     prev_img.shape[1],
    #     3
    # ), np.float32)

    # np.average()

    # #Setting the hue to be the direction of movement
    # mask[..., 0] = angle*180 / np.pi / 2
    # #Setting saturation to 100%
    # mask[..., 1] = 255
    # #Setting the value to be the magnitude of movement
    # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_L1)

# plt.scatter(magnitudes, angles, c=range(len(angles)), marker="o")
# plt.ylabel("Angle")
# plt.xlabel("Magnitude")
# plt.show()