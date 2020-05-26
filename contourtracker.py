from TimeSlice import TimeSlice

import cv2
import math

def euclid_dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def center(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w/2, y + h/2)

class ContourTracker():
    def __init__(self, proximity_thresh, time_thresh, min_duration):
        self.proximity_thresh = proximity_thresh**2
        self.time_thresh = time_thresh
        self.min_duration = min_duration
        self.contours = []

    def add_contour(self, new_contour, time):
        closest_contour = (None, 1000)

        for contour in self.contours:
            if contour.is_active(time):
                distance= euclid_dist(
                    center(contour.get_recent()),
                    center(new_contour)
                )

                if distance < closest_contour[1]:
                    closest_contour = (contour, distance)

        if closest_contour[1] <= self.proximity_thresh:
            closest_contour[0].add(new_contour, time)
        else:
            self.contours.append(
                TimeContour(
                    len(self.contours),
                    new_contour, time,
                    self)
                )

    def get_json(self, s):
        json = dict();

        for tcontour in self.contours:
            if tcontour.is_dead(): continue #Skip iteration if contour is dead

            for contour in tcontour.contours.items():
                x, y, w, h = cv2.boundingRect(contour[1])

                if not contour[0] in json.keys():
                    json[contour[0]] = { tcontour.id :  [x*s, y*s, w*s, h*s]}

                json[contour[0]][tcontour.id] = [x*s, y*s, w*s, h*s]

        return {"rois":json};

    def get_timeslices(self, framerate):
        frametime = 1000 / framerate
        timeslices = []

        sorted_contours = list(sorted(self.contours, key=lambda cntr: cntr.start))

        print(len(sorted_contours))

        contour = sorted_contours.pop(0)

        start = contour.start
        recent = contour.recent

        while len(sorted_contours) > 0:
            contour = sorted_contours.pop(0)

            if contour.is_dead(): continue #Skip iteration if contour is dead

            if contour.start <= start:
                recent = contour.recent
            else:
                timeslices.append(TimeSlice(
                    start * frametime,
                    (recent-start) * frametime
                ))

                if len(sorted_contours) > 0:
                    contour = sorted_contours.pop(0)
                    start = contour.start
                    recent = contour.recent

        timeslices.append(TimeSlice(
                    start * frametime,
                    (recent-start) * frametime
                ))

        print(len(timeslices))

        return timeslices

    def draw_countours(self, image, scale, time):
        for contour in self.contours:
            if contour.is_dead(): continue #Skip iteration if contour is dead
            if contour.start <= time and contour.recent >= time:
                image = contour.draw(image, scale, time)
        return image

class TimeContour():
    def __init__(self, id, contour, start, parent):
        self.id = id
        self.contours = {start:contour}
        self.parent = parent
        self.start = start
        self.recent = start

    def is_active(self, time):
        return (time - self.recent) <= self.parent.time_thresh and self.recent != time 

    def is_dead(self):
        return self.recent-self.start < self.parent.min_duration

    def get_recent(self):
        return self.contours[self.recent]

    def get_closest(self, time):
        safe_time = time

        while safe_time not in self.contours.keys():
            safe_time-=1

        return self.contours[safe_time]

    def draw(self, image, scale, time):
        x, y, w, h = cv2.boundingRect(self.get_closest(time))

        x = math.floor(x * scale[0])
        y = math.floor(y * scale[1])
        w = math.floor(w * scale[0])
        h = math.floor(h * scale[1])
        
        img = cv2.rectangle( image, (x, y), (x+w, y+h), (0,0,255), 1 )
        return cv2.putText(img, str(self.id), (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25*scale[1], (255,255,255))

    def add(self, contour, time):
        self.contours[time] = contour
        self.recent = time