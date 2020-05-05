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

    def intersection_over_union(self, other):
        intersection = min(self.end(), other.end()) - max(self.start, other.start)
        union = max(self.end(), other.end()) - min(self.start, other.start)

        print(max(self.end(), other.end()))
        print(str(union))

        return intersection / union
