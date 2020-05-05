import cv2

def get_anomalies(gm_model, path, feature_function, threshold):
    framebuffer = cv2.VideoCapture(path)
    features = []

    if not framebuffer.isOpened():
        print('Failed to open path: ' + str(path))

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

    features_proba = enumerate(
        gm_model.predict_proba(features)[:, 1]
    )

    anomalies = list(
        map(
            lambda anomaly: anomaly[0],
            filter(
                lambda anomaly_proba: anomaly_proba[1] >= threshold,
                features_proba
            )
        )
    )

    return sorted(anomalies)
