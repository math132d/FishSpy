import cv2
import math

def euclid_dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def center(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return [x + w/2, y + h/2]

class ContourTracker():
    def __init__(self, proximity_thresh, time_thresh, min_duration):
        self.proximity_thresh = proximity_thresh**2
        self.time_thresh = time_thresh
        self.min_duration = min_duration
        self.contours = []

    def add_contour(self, new_contour, time):
        closest_contour = (None, 1000)

        for contour in self.contours:
            if not contour.is_dead(time):
                distance = euclid_dist( contour.get_recent_center(), center(new_contour) )

                if (distance < closest_contour[1] and
                    time-contour.recent <= self.time_thresh):
                    closest_contour = (contour, distance)
            else:
                self.contours.remove(contour)

        if closest_contour[1] < self.proximity_thresh:
            closest_contour[0].add(new_contour, time)
        else:
            self.contours.append(TimeContour(len(self.contours), new_contour, time, self))

    def draw_countours(self, image, time):
        for contour in self.contours:
            if contour.start <= time and contour.recent >= time:
                image = contour.draw(image, time)
        return image

class TimeContour():
    def __init__(self, id, contour, start, parent):
        self.id = id
        self.contours = {start:contour}
        self.parent = parent
        self.start = start
        self.recent = start

    def is_dead(self, frame):
        return frame-self.recent > self.parent.time_thresh and self.recent-self.start < self.parent.min_duration

    def get_recent_center(self):
        return center(self.contours[self.recent])

    def draw(self, image, time):
        safe_time = time

        while safe_time not in self.contours.keys():
            safe_time-=1

        x, y, w, h = cv2.boundingRect(self.contours[safe_time])
        img = cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 1)
        return cv2.putText(img, str(self.id), (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255))

    def add(self, contour, time):
        self.contours[time] = contour
        self.recent = time