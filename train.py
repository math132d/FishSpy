from sklearn.mixture import GaussianMixture

import cv2
import utils

def train(path):
    GM = GaussianMixture(n_components=2)
    FRAMEBUFFER = cv2.VideoCapture(path)
    FEATURES = []

    if not FRAMEBUFFER.isOpened():
        print('Failed to open video file')

    _ret, prev_frame = FRAMEBUFFER.read()
    _ret, curr_frame = FRAMEBUFFER.read()

    while FRAMEBUFFER.isOpened():

        if _ret == False:
            print('Reached end of file')
            break

        feature = utils.mean_squared_error(
            prev_frame,
            curr_frame
        )

        prev_frame = curr_frame
        _ret, curr_frame = FRAMEBUFFER.read()

        FEATURES.append([feature])

    FRAMEBUFFER.release()
    GM.fit(FEATURES)

    return GM
