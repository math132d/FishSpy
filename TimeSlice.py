from datetime import timedelta

class TimeSlice:
    def __init__(self, start, duration):
        #Start and duration are both in milliseconds
        self.start = start
        self.duration = duration

    def __str__(self):
        return self.get_timestamp()

    def end(self):
        return self.start + self.duration

    def get_timestamp(self):
        start = timedelta(milliseconds=self.start)
        end = timedelta(milliseconds=self.end())

        return "[{} -> {}]".format(str(start), str(end))
