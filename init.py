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

    image1, image2 = utils.load_images(
        path.join(FRAMES_PATH, FRAMES_LIST[idx-1]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx])    
    )

    #Change this function to change how the difference is calculated
    change = utils.optical_flow_field( image1, image2 )

    #Append as 1-element arrays since GM expects a 2D array
    MSE_LIST.append([change[0], change[1]])

GM.fit(MSE_LIST)

MSE_LIST_PROBA = enumerate(GM.predict_proba(MSE_LIST)[:, 1])

ANOMS = list(
    map(
        lambda anomal: anomal[0],
        filter(
            lambda anomal_proba: anomal_proba[1] >= 0.8,
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
