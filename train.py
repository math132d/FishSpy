from sklearn.mixture import GaussianMixture

import cv2
import json
import utils

GM = GaussianMixture(n_components=2)
FRAMEBUFFER = cv2.VideoCapture('./vid/train_01_sm.mp4')
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

GM.fit(FEATURES)

params = GM.get_params(deep=True)

with open('./params.json', 'w') as output:
    json.dump(params, output)
