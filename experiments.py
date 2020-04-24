from os import path
from skimage.measure import compare_ssim

import cv2
import numpy as np

import utils

FRAMES_PATH = path.abspath("./vid/frames/")
FRAMES_LIST = utils.files_from(FRAMES_PATH)

for idx in range(len(FRAMES_LIST)):
    (IMG1, IMG2) = utils.load_images(
        path.join(FRAMES_PATH, FRAMES_LIST[idx+1]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx])
    )

    (IMG1_ALT, IMG2_ALT) = utils.detect_edges((IMG1, IMG2))

    # _, IMG1_ALT = cv2.threshold(IMG1_ALT, 15, 255, cv2.THRESH_BINARY)
    # _, IMG2_ALT = cv2.threshold(IMG2_ALT, 15, 255, cv2.THRESH_BINARY)

    # DISP = np.vstack((
    #         np.hstack((IMG1, IMG2)),
    #         np.hstack((IMG1_ALT, IMG2_ALT))
    #     ))

    DIFF = cv2.subtract(IMG1, IMG2)
    DIFF_ALT = cv2.subtract(IMG1_ALT, IMG2_ALT)
    (_, DIFF_SSIM) = compare_ssim(IMG1, IMG2, full=True)

    _, DIFF = cv2.threshold(DIFF, 10, 255, cv2.THRESH_BINARY)
    _, DIFF_EDGE = cv2.threshold(DIFF_ALT, 10, 255, cv2.THRESH_BINARY)

    DIFFF = np.hstack((DIFF, DIFF_SSIM))

    # cv2.imshow("Display",  DISP)
    # cv2.waitKey(0)
    cv2.imshow("Display", DIFFF)
    cv2.waitKey(0)