from sklearn.mixture import GaussianMixture
import cv2

def train(path, feature_function):
    gm = GaussianMixture(n_components=2)
    framebuffer = cv2.VideoCapture(path)
    features = []

    if not framebuffer.isOpened():
        print('Failed to open video file')

    _ret, prev_frame = framebuffer.read()
    _ret, curr_frame = framebuffer.read()

    while framebuffer.isOpened():

        if not _ret:
            print('Reached end of file')
            break

        feature = feature_function(
            prev_frame,
            curr_frame
        )

        prev_frame = curr_frame
        _ret, curr_frame = framebuffer.read()

        features.append([feature])

    framebuffer.release()
    gm.fit(features)

    return gm
