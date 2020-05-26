from sklearn.mixture import GaussianMixture

import cv2
import utils

def train(path, feature_function):
    gm = GaussianMixture(n_components=2)
    framebuffer = cv2.VideoCapture(path)
    features = []

    if not framebuffer.isOpened():
        print('Failed to open video file')

    _ret, prev_frame = utils.read_grayscale(framebuffer)
    _ret, curr_frame = utils.read_grayscale(framebuffer)

    while framebuffer.isOpened():

        if not _ret:
            print('Reached end of file')
            break

        feature = feature_function(
            prev_frame,
            curr_frame
        )

        prev_frame = curr_frame
        _ret, curr_frame = utils.read_grayscale(framebuffer)

        features.append(feature)

    framebuffer.release()
    gm.fit(features)

    return gm
