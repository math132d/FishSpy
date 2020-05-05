import matplotlib.pyplot as plt
import numpy as np

from train import train
from anomalies import get_anomalies

import utils

GM = train('./vid/train_01.mp4')

ANOMS = get_anomalies(GM, './vid/train_01.mp4', 0.98)

ANOMS = utils.filter_frames(ANOMS, 13)

ANOMS_VECTOR = np.zeros(np.max(ANOMS), np.uint8)

for ANOM in ANOMS:
    ANOMS_VECTOR[ANOM-1] = 1

time_slices = utils.get_time_slices(ANOMS, 25, 25)

for s in time_slices:
    print(s.get_timestamp())

plt.plot(range(len(ANOMS_VECTOR)), ANOMS_VECTOR, 'x')
plt.show()
