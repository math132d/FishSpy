from os import path, listdir

from sklearn.mixture import GaussianMixture

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(path1, path2):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    err = np.sum((image1.astype("float")-image2.astype("float")) ** 2)
    return err / (image1.shape[0] * image1.shape[1])

FRAMES_PATH = path.abspath("./vid/frames/")

print(FRAMES_PATH)

FRAMES_LIST = list(filter( #Include only entries that are files, not folders
    lambda file: path.isfile(path.join(FRAMES_PATH, file)),
    listdir(FRAMES_PATH)
))

#Sort frames to be in order (linux dont return files in logial order...)
FRAMES_LIST.sort()

MSE_LIST = []

GM = GaussianMixture(n_components=2)

print("Looking for frames in: " + FRAMES_PATH)

for idx in range(len(FRAMES_LIST)-1):
    mean_error = mean_squared_error(
        path.join(FRAMES_PATH, FRAMES_LIST[idx-1]),
        path.join(FRAMES_PATH, FRAMES_LIST[idx])
    )

    #Append as 1-element arrays since GM expects a 2D array
    MSE_LIST.append([mean_error])

GM.fit(MSE_LIST)

#MSE_LIST_PROBA = enumerate(GM.predict_proba(MSE_LIST)[:,1])

# ANOMALTIES = sorted(
#     MSE_LIST_PROBA,
#     key=lambda x: x[1]
# )[:10]

# print(ANOMALTIES)

plt.figure(figsize=(8,6))
#Plot MSE difference
plt.scatter(range(len(MSE_LIST)), MSE_LIST,c=GM.predict_proba(MSE_LIST)[:,1], marker="x")
#Plot probability
#plt.scatter(range(len(MSE_LIST)), GM.predict_proba(MSE_LIST)[:,1], marker="x")
plt.show()