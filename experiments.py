import cv2
import numpy as np
import utils

path = './vid/small.mp4'
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

framebuffer = cv2.VideoCapture(path)
kernel_sm = np.ones((3,3), np.uint8)
kernel_lg = np.ones((6,6), np.uint8)

if not framebuffer.isOpened():
    print('Failed to open video file')

i=0

while framebuffer.isOpened():
    i += 1;

    _ret, curr_frame = framebuffer.read()

    if not _ret:
        print('Reached end of file')
        break

    fg = bg_subtractor.apply(curr_frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY)
    fg = cv2.dilate(fg, kernel_sm, iterations=2)

    ctr, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE);

    cv2.drawContours(curr_frame, ctr, -1, (0,255,0))

    # curr_frame[:,:,2] = curr_frame[:,:,2] * 0.5 + (fg * 0.5)
    # cv2.imshow("Frame", curr_frame)
    cv2.imwrite(f'./vid/frames/frame_{i}.png', curr_frame)
    # cv2.waitKey(0)