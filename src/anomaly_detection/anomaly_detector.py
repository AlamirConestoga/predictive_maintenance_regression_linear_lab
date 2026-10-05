# src/anomaly_detection/anomaly_detector.py

class AnomalyDetector:
    """
    Stateful streaming anomaly detector for one axis.

    Rules (from the lab spec):
      - ALERT  : residual >= min_c for >= time_threshold seconds continuously
      - ERROR  : residual >= max_c for >= time_threshold seconds continuously

    Design:
      - A "deviation" begins the first time residual >= min_c.
      - While the deviation persists, the detector tracks the maximum residual.
      - If the residual reaches max_c during a deviation, the event is
        escalated from ALERT to ERROR.
      - Once the event has been fired (after T seconds), the detector does
        NOT fire again for the same deviation. It only re-arms when the
        residual drops back below min_c.
      - This guarantees one continuous deviation -> one event.
    """

    def __init__(self, min_c, max_c, time_threshold, axis="unknown"):
        if min_c <= 0 or max_c <= 0:
            raise ValueError("min_c and max_c must be positive")
        if max_c <= min_c:
            raise ValueError("max_c must be greater than min_c")
        if time_threshold <= 0:
            raise ValueError("time_threshold must be positive")

        self.min_c = min_c
        self.max_c = max_c
        self.time_threshold = time_threshold
        self.axis = axis

        # --- State for the currently active deviation ---
        self.event_start = None
        self.event_type = None
        self.max_deviation = None
        self.event_fired = False

        # --- History of finalised events ---
        self.events = []

    # ------------------------------------------------------------------
    def update(self, timestamp, actual, predicted):
        """
        Process one observation.

        Returns:
            dict -> an event was finalised on this call
            None -> no event finalised yet
        """
        residual = actual - predicted

        # --- Determine current level --------------------------------
        if residual >= self.max_c:
            level = "error"
        elif residual >= self.min_c:
            level = "alert"
        else:
            level = None

        # --- Case 1: residual dropped below min_c -------------------
        # The deviation (if any) has ended. Reset state.
        if level is None:
            self._reset_state()
            return None

        # --- Case 2: first sample of a new deviation ----------------
        if self.event_start is None:
            self.event_start = timestamp
            self.event_type = level
            self.max_deviation = residual
            self.event_fired = False
            return None

        # --- Case 3: deviation continues but event already fired ----
        if self.event_fired:
            if residual > self.max_deviation:
                self.max_deviation = residual
            return None

        # --- Case 4: deviation continues, event not yet fired -------
        if level == "error" and self.event_type == "alert":
            self.event_type = "error"

        if residual > self.max_deviation:
            self.max_deviation = residual

        elapsed = (timestamp - self.event_start).total_seconds()
        if elapsed >= self.time_threshold:
            event = self._finalise_event(timestamp)
            self.event_fired = True
            return event

        return None

    # ------------------------------------------------------------------
    def _finalise_event(self, end_timestamp):
        """Build the event dict, store it, and return it."""
        duration = (end_timestamp - self.event_start).total_seconds()
        event = {
            "axis": self.axis,
            "type": self.event_type,
            "start": self.event_start,
            "end": end_timestamp,
            "duration_seconds": duration,
            "max_deviation": self.max_deviation,
        }
        self.events.append(event)
        return event

    def _reset_state(self):
        self.event_start = None
        self.event_type = None
        self.max_deviation = None
        self.event_fired = False