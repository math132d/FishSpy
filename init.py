from os import path

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import utils

FRAMES_PATH = path.abspath("./vid/frames/")
FRAMES_LIST = utils.files_from(FRAMES_PATH)

MSE_LIST = []
GM = GaussianMixture(n_components=2)

print("Looking for frames in: " + FRAMES_PATH)
for idx in range(len(FRAMES_LIST)-1):
    mean_error = utils.mean_squared_error(
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