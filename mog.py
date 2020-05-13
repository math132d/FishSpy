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

    # if(frame in [2079, 3355, 3221, 2903, 1490]):
    #     cv2.waitKey(0)
    #     ctr_img = contourtracker.draw_countours(curr_frame, frame)
    #     cv2.imshow("frame", ctr_img)

    # curr_frame[:,:,2] = curr_frame[:,:,2] * 0.5 + (fg * 0.5)
    # cv2.imshow("Frame", curr_frame)
    ctr_img = contourtracker.draw_countours(curr_frame, frame)
    cv2.imwrite(f'./vid/frames/frame_{frame}.png', ctr_img)
    # cv2.waitKey(0)