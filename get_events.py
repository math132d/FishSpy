from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import numpy as np
import cv2
import json

import TimeSlice
import utils

with open('./params.json') as param_file:
    params = json.loads(param_file.read())

GM = GaussianMixture(n_components=2).set_params(**params)
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

FEATURES_PROBA = enumerate(GM.predict_proba(FEATURES)[:, 1])

ANOMS = list(
    map(
        lambda anomal: anomal[0],
        filter(
            lambda anomal_proba: anomal_proba[1] >= 0.98,
            FEATURES_PROBA
        )
    )
)

ANOMS = sorted(ANOMS)

print(ANOMS)

ANOMS = utils.filter_frames(ANOMS, 13)

print(ANOMS)

ANOMS_VECTOR = np.zeros(np.max(ANOMS), np.uint8)

for ANOM in ANOMS:
    ANOMS_VECTOR[ANOM-1] = 1

time_slices = utils.get_time_slices(ANOMS, 25, 25)

for s in time_slices:
    print(s.get_timestamp())

plt.plot(range(len(ANOMS_VECTOR)), ANOMS_VECTOR, 'x')
plt.show()