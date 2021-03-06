import cv2
import utils

def get_anomalies(gm_model, path, feature_function):
    framebuffer = cv2.VideoCapture(path)
    features = []

    if not framebuffer.isOpened():
        print('Failed to open path: ' + str(path))

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

    features_proba = enumerate(
        gm_model.predict_proba(features)[:, 1]
    )

    return features_proba
