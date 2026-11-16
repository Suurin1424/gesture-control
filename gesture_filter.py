class GestureFilter:
    """Zamienia strumien gestow z kolejnych klatek na pojedyncze zdarzenia.

    Gest wyzwala zdarzenie, gdy utrzyma sie przez stable_frames klatek.
    Potem jest zablokowany, dopoki przez release_frames klatek nie zostanie
    pokazane cos innego (inny gest albo brak dloni). Dzieki temu trzymanie
    kciuka w gorze przewija jeden slajd, a nie trzydziesci.
    """

    def __init__(self, stable_frames: int, release_frames: int, min_score: float):
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.min_score = min_score
        self._candidate = None   # gest obserwowany w kolejnych klatkach
        self._count = 0          # ile klatek z rzedu trwa _candidate
        self._latched = None     # ostatnio wyzwolony gest (zablokowany)
        self._release_count = 0  # ile klatek z rzedu nie ma _latched

    def update(self, gesture: str | None, score: float) -> str | None:
        """Przyjmuje gest z jednej klatki. Zwraca nazwe gestu, gdy nalezy wykonac akcje."""
        if gesture == "None" or score < self.min_score:
            gesture = None

        # Zdejmowanie blokady: zablokowany gest musi zniknac na release_frames klatek.
        if self._latched is not None:
            if gesture == self._latched:
                self._release_count = 0
            else:
                self._release_count += 1
                if self._release_count >= self.release_frames:
                    self._latched = None
                    self._release_count = 0

        # Stabilnosc: liczymy, ile klatek z rzedu trwa ten sam gest.
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1

        if (gesture is not None
                and gesture != self._latched
                and self._count >= self.stable_frames):
            self._latched = gesture
            self._release_count = 0
            return gesture
        return None

    @property
    def progress(self) -> float:
        """Postep 0..1 do wyzwolenia biezacego gestu (0, gdy brak gestu albo gest zablokowany)."""
        if self._candidate is None or self._candidate == self._latched:
            return 0.0
        return min(self._count / self.stable_frames, 1.0)

    @property
    def latched(self) -> str | None:
        """Gest, ktory juz wyzwolil akcje i czeka na zwolnienie."""
        return self._latched