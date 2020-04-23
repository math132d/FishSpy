from os import path

from sklearn.mixture import GaussianMixture
from TimeSlice import TimeSlice

import matplotlib.pyplot as plt
import cv2
import utils

FRAMES_PATH = path.abspath("./vid/frames/")
FRAMES_LIST = utils.files_from(FRAMES_PATH)

MSE_LIST = []
GM = GaussianMixture(n_components=2)

print("Looking for frames in: " + FRAMES_PATH)
for idx in range(1, len(FRAMES_LIST)):
    mean_error = utils.mean_squared_error( #Change this function to change how the difference is calculated
        path.join(FRAMES_PATH, FRAMES_LIST[idx-1]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx])
    )

    #Append as 1-element arrays since GM expects a 2D array
    MSE_LIST.append([mean_error])

GM.fit(MSE_LIST)

MSE_LIST_PROBA = enumerate(GM.predict_proba(MSE_LIST)[:, 1])

ANOMALIES = list(
    map(
        lambda anomal: anomal[0],
        filter(
            lambda anomal_proba: anomal_proba[1] >= 0.98,
            MSE_LIST_PROBA
        )
    )
)

time_slices = sorted(utils.get_time_slices(ANOMALIES, 25), key=lambda x: x.duration);

for sl in time_slices:
    print(sl.get_timestamp())

# ANOMALTIES = sorted(
#     MSE_LIST_PROBA,
#     key=lambda x: x[1],
#     reverse=True
# )

# for anom in ANOMALTIES:
#     if anom[1] < 0.8:
#         break

#     print(anom[0])

fig, ax_diff = plt.subplots()

ax_diff.scatter(range(len(MSE_LIST)), MSE_LIST, c=GM.predict_proba(MSE_LIST)[:, 1], marker="x")

ax_proba = ax_diff.twinx()

ax_proba.scatter(range(len(MSE_LIST)), GM.predict_proba(MSE_LIST)[:, 1], marker="x")

fig.tight_layout()
plt.show()
