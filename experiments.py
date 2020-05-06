from TimeSlice import TimeSlice
from eval import evaluate_iou

ground_truth = [
    TimeSlice(0, 200),
    TimeSlice(6000, 600),
    TimeSlice(8000, 600),
]

inferred = [
    TimeSlice(250, 200),
    TimeSlice(5000, 600),
    TimeSlice(6200, 200),
    TimeSlice(8000, 600),
    TimeSlice(10000, 200)
]

print(
    evaluate_iou(ground_truth, inferred, 0.2)
)