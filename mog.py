import cv2
import numpy as np
import utils

from contourtracker import ContourTracker

path = './vid/small.mp4'
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=750)
contourtracker = ContourTracker(25, 3, 10)

framebuffer = cv2.VideoCapture(path)

kernel_sm = np.ones((3,3), np.uint8)
kernel_lg = np.ones((6,6), np.uint8)

if not framebuffer.isOpened():
    print('Failed to open video file')

frame=0
ctr_img = None

while framebuffer.isOpened():
    frame += 1;

    _ret, curr_frame = framebuffer.read()

    if not _ret:
        print('Reached end of file')
        break

    fg = bg_subtractor.apply(curr_frame)

    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY)
    fg = cv2.dilate(fg, kernel_sm, iterations=4)

    contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);

    for contour in contours:
        contourtracker.add_contour(contour, frame)

framebuffer = cv2.VideoCapture(path)
frame=0

if not framebuffer.isOpened():
    print('Failed to open video file again')

while framebuffer.isOpened():
    frame += 1;
    _ret, curr_frame = framebuffer.read()

    if not _ret:
        print('Reached end of file')
        break

    img = contourtracker.draw_countours(curr_frame, frame)

    cv2.imwrite(f'/vid/frames/frame_{frame}.png', curr_frame)