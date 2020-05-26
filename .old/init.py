import matplotlib.pyplot as plt
import numpy as np

from train import train
from TimeSlice import TimeSlice
from anomalies import get_anomalies
from eval import evaluate_iou

import utils

ground_truth = [
    TimeSlice(95000, 5000),
    TimeSlice(121000, 6000),
    TimeSlice(148000, 4000),
    TimeSlice(337000, 10000),
    TimeSlice(360000, 6000),
    TimeSlice(374000, 11000),
    TimeSlice(410000, 14000),
    TimeSlice(431000, 6000),
    TimeSlice(542000, 6000)
]

GM = train('./vid/test_train_02_sm.mp4', utils.mean_squared_error)

features_proba = get_anomalies(GM, './vid/test_train_02_sm.mp4', utils.mean_squared_error)

for i in range(5):
    threshold = 1.0 - (0.05 * i)
    inferred = utils.anoms_to_timeslice(features_proba, threshold)
    result = evaluate_iou(ground_truth, inferred, 0.5)

    print(str(threshold) + ":" + str(result))

# ANOMS = utils.filter_frames(ANOMS, 13)

# ANOMS_VECTOR = np.zeros(np.max(ANOMS), np.uint8)

# for ANOM in ANOMS:
#     ANOMS_VECTOR[ANOM-1] = 1

# time_slices = utils.get_time_slices(ANOMS, 25, 25)

# for s in time_slices:
#     print(s.get_timestamp())

# plt.plot(range(len(ANOMS_VECTOR)), ANOMS_VECTOR, 'x')
# plt.show()