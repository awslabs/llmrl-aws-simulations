class LinearScheduler(object):
    def __init__(self, start: float, end: float, total_steps: int):
        assert total_steps > 0, "Total steps must be greater than 0."
        self._start = start
        self._end = end
        self._total_steps = total_steps
        self._slope = (self._end - self._start) / self._total_steps
        self._steps = 0

    def update(self):
        self._steps += 1

    def __call__(self):
        if self._steps >= self._total_steps:
            return self._end
        else:
            return self._slope * self._steps + self._start

    def state_dict(self):
        return {
            "start": self._start,
            "end": self._end,
            "total_steps": self._total_steps,
            "steps": self._steps,
        }

    def load_from_state_dict(self, state):
        self._start = state["start"]
        self._end = state["end"]
        self._total_steps = state["total_steps"]
        self._steps = state["steps"]
        self._slope = (self._end - self._start) / self._total_steps
