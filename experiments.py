import mog
import json

from TimeSlice import TimeSlice
from eval import evaluate_iou

ground_truth = [
    TimeSlice(14200, 8800),
    TimeSlice(51400, 6640),
    TimeSlice(79200, 2800),
    TimeSlice(119400, 2840),
    TimeSlice(137240, 4720),
    TimeSlice(233360, 62600),
    TimeSlice(354000, 7520),
    TimeSlice(455080, 2200),
    TimeSlice(513560, 2480),
    TimeSlice(536800, 1800),
    TimeSlice(571040, 2040),
    TimeSlice(576720, 3000)
 ]

contourtracker = mog.get_tracked_contours('./vid/clip_3.mp4', space_grouping_threshold=10, time_grouping_threshold=3, min_duration=25)

with open('clip_3_lg.json', 'w') as file:
    json.dump(contourtracker.get_json(6), file)

# contourtracker = mog.get_tracked_contours('./vid/test_small.mp4', space_grouping_threshold=10, time_grouping_threshold=9, min_duration=25)

# mog.draw_tracked_contours('./vid/small.mp4', './vid/test/', contourtracker)

# inferred = contourtracker.get_timeslices(25);

# for tslice in inferred:
#     print(
#         tslice.get_timestamp()
#     )

# tp, fp, fn, precision, recall = evaluate_iou(ground_truth, inferred, 0.5)

# print(f'{tp}:{fp}:{fn}:{precision}:{recall}');


# ground_truth = [
#     TimeSlice(39000, 2000),
#     TimeSlice(82000, 1440),
#     TimeSlice(122000, 1560),
#     TimeSlice(136000, 1920),
#     TimeSlice(142000, 1800),
#     TimeSlice(150000, 1720),
#     TimeSlice(241000, 6320),
#     TimeSlice(249000, 1720),
#     TimeSlice(273000, 1040),
#     TimeSlice(276000, 2680),
#     TimeSlice(295000, 2000),
#     TimeSlice(305000, 1040),
#     TimeSlice(322000, 3200),
#     TimeSlice(459000, 2200),
#     TimeSlice(469000, 3040),
# ]

# for prxthresh in range(10, 40, 10):
#     for tmthresh in range(3, 15, 3):
#         for mindur in range(5, 30, 5):
#             contourtracker = mog.get_tracked_contours('./vid/train_small.mp4', space_grouping_threshold=prxthresh, time_grouping_threshold=tmthresh, min_duration=mindur)
#             inferred = contourtracker.get_timeslices(25);
#             tp, fp, fn, precision, recall = evaluate_iou(ground_truth, inferred, 0.5)

#             print(f'{prxthresh}:{tmthresh}:{mindur} - {tp}:{fp}:{fn}:{precision}:{recall}');