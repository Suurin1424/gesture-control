from collections import deque

WRIST = 0  # indeks nadgarstka w 21 punktach dloni MediaPipe


class SwipeDetector:
    """Wykrywa machniecie dlonia w lewo/prawo na podstawie ruchu nadgarstka."""

    def __init__(self, window_s: float, min_distance: float, max_vertical_ratio: float,
                 cooldown_s: float, required_gesture: str | None = None):
        self.window_s = window_s                      # okno czasowe ruchu [s]
        self.min_distance = min_distance              # min. przesuniecie poziome (ulamek szerokosci kadru)
        self.max_vertical_ratio = max_vertical_ratio  # max |dy| / |dx| - ruch ma byc glownie poziomy
        self.cooldown_s = cooldown_s                  # blokada po wykryciu (powrot reki)
        self.required_gesture = required_gesture      # np. "Open_Palm"; None = dowolny ksztalt dloni
        self._history = deque()                       # (czas_s, x, y)
        self._blocked_until = 0.0

    def update(self, landmarks, gesture: str | None, t: float) -> str | None:
        """landmarks: 21 punktow pierwszej dloni albo None; t: czas w sekundach.
        Zwraca "Swipe_Right" / "Swipe_Left" albo None."""
        if landmarks is None:
            self._history.clear()  # dlon zniknela - ruch sie urwal
            return None

        if t < self._blocked_until:
            self._history.clear()  # w trakcie blokady nie zbieramy ruchu powrotnego
            return None

        if self.required_gesture is not None and gesture != self.required_gesture:
            return None  # klatka bez wymaganego ksztaltu dloni jest pomijana

        wrist = landmarks[WRIST]
        self._history.append((t, wrist.x, wrist.y))
        while t - self._history[0][0] > self.window_s:
            self._history.popleft()  # usuwamy punkty starsze niz okno

        if len(self._history) < 2:
            return None

        _, x0, y0 = self._history[0]
        dx = wrist.x - x0
        dy = abs(wrist.y - y0)
        if abs(dx) >= self.min_distance and dy <= self.max_vertical_ratio * abs(dx):
            self._history.clear()
            self._blocked_until = t + self.cooldown_s
            return "Swipe_Right" if dx > 0 else "Swipe_Left"
        return None

    @property
    def trail(self) -> list[tuple[float, float]]:
        """Punkty (x, y) ruchu nadgarstka z biezacego okna czasowego."""
        return [(x, y) for _, x, y in self._history]

    @property
    def progress(self) -> float:
        """Postep 0..1: przesuniecie poziome wzgledem wymaganego min_distance."""
        if len(self._history) < 2:
            return 0.0
        _, x0, _ = self._history[0]
        _, x1, _ = self._history[-1]
        return min(abs(x1 - x0) / self.min_distance, 1.0)

    def is_cooling_down(self, t: float) -> bool:
        return t < self._blocked_until