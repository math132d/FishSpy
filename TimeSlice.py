from datetime import timedelta

class TimeSlice:
    def __init__(self, start, duration):
        self.start = start
        self.duration = duration

    def end(self):
        return self.start + self.duration

    def get_timestamp(self):
        start = timedelta(milliseconds=self.start * (1000/25))
        end = timedelta(milliseconds=self.end() * (1000/25))

        return "[{} -> {}]".format(str(start), str(end))
