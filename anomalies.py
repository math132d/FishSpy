import cv2
import utils

def get_anomalies(gm_model, path, threshold):
    FRAMEBUFFER = cv2.VideoCapture(path)
    FEATURES = []

    if not FRAMEBUFFER.isOpened():
        print('Failed to open path: ' + str(path))

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

    FEATURES_PROBA = enumerate(
        gm_model.predict_proba(FEATURES)[:, 1]
    )

    anomalies = list(
        map(
            lambda anomaly: anomaly[0],
            filter(
                lambda anomaly_proba: anomaly_proba[1] >= threshold,
                FEATURES_PROBA
            )
        )
    )

    return sorted(anomalies)
