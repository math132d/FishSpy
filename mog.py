from os import path
import cv2
import numpy as np
import utils

from contourtracker import ContourTracker

def get_tracked_contours(in_path, space_grouping_threshold=10, time_grouping_threshold=3, min_duration=10, history=750):
    #Function returns a contourtracker object which contains a list of TimeCountours (tracked contours)

    contourtracker = ContourTracker(time_grouping_threshold, time_grouping_threshold, min_duration)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history)
    framebuffer = cv2.VideoCapture(in_path)
    kernel_sm = np.ones((3,3), np.uint8)

    if not framebuffer.isOpened():
        print('Failed to open video file')

    frame=0
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

    return contourtracker

def draw_tracked_contours(in_path, out_path, contourtracker, scale=(1.0, 1.0)):
    #in_path: path to a video
    #contourtracker: object with pre tracked contours
    #scale: A touple of (x, y) scales at which to draw the bounding boxes on the video (if the contourtracker tracked a different size video)
    #takes a path to a video and contourtracker and draws the bounding box of the tracked contours on the video

    framebuffer = cv2.VideoCapture(in_path)
    
    frame=0
    if not framebuffer.isOpened():
        print('Failed to open video file again')

    while framebuffer.isOpened():
        frame += 1;
        _ret, curr_frame = framebuffer.read()

        if not _ret:
            print('Reached end of file')
            break

        img = contourtracker.draw_countours(curr_frame, scale, frame)

        cv2.imwrite(path.join(out_path, f'frame_{frame}.png'), curr_frame)