from os import path

from sklearn.mixture import GaussianMixture
from TimeSlice import TimeSlice

import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils

FRAMES_PATH = path.abspath("./vid/frames/")
FRAMES_LIST = utils.files_from(FRAMES_PATH)

MSE_LIST = []
GM = GaussianMixture(n_components=2)

print("Looking for frames in: " + FRAMES_PATH)
for idx in range(1, len(FRAMES_LIST)):
    #Change this function to change how the difference is calculated
    diff = utils.optical_flow_field(
        path.join(FRAMES_PATH, FRAMES_LIST[idx-1]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx])
    )

    #Append as 1-element arrays since GM expects a 2D array
    MSE_LIST.append([diff[0], diff[1]])

GM.fit(MSE_LIST)

MSE_LIST_PROBA = enumerate(GM.predict_proba(MSE_LIST)[:, 1])

ANOMS = list(
    map(
        lambda anomal: anomal[0],
        filter(
            lambda anomal_proba: anomal_proba[1] >= 0.98,
            MSE_LIST_PROBA
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

# fig, ax_diff = plt.subplots()

# ax_diff.scatter(range(len(MSE_LIST)), MSE_LIST, c=GM.predict_proba(MSE_LIST)[:, 1], marker="x")

# ax_proba = ax_diff.twinx()

# ax_proba.scatter(range(len(MSE_LIST)), GM.predict_proba(MSE_LIST)[:, 1], marker="o")

# fig.tight_layout()
# plt.show()
