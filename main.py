import mog

INPUT_VIDEO = './vid/test_small.mp4'
OUTPUT_PATH = './vid/output/'

contourtracker = mog.get_tracked_contours(INPUT_VIDEO, space_grouping_threshold=10, time_grouping_threshold=9, min_duration=25)

mog.draw_tracked_contours(INPUT_VIDEO, OUTPUT_PATH, contourtracker)