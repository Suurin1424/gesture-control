import cv2

# Kolory w BGR (kolejnosc OpenCV)
GRAY = (160, 160, 160)
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
BLUE = (255, 150, 0)
WHITE = (255, 255, 255)

# Zalecany obszar dla nadgarstka (znormalizowane x1, y1, x2, y2)
ZONE = (0.2, 0.2, 0.8, 0.9)
FLASH_S = 0.6  # jak dlugo ramka swieci na niebiesko po wykonaniu akcji
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(frame, x: float, y: float) -> tuple[int, int]:
    """Wspolrzedne znormalizowane -> piksele."""
    h, w = frame.shape[:2]
    return int(x * w), int(y * h)


def _bar(frame, y: int, value: float, color, label: str) -> None:
    w = frame.shape[1]
    x1, x2 = 10, w - 10
    cv2.rectangle(frame, (x1, y), (x2, y + 14), GRAY, 1)
    cv2.rectangle(frame, (x1, y), (x1 + int((x2 - x1) * value), y + 14), color, -1)
    cv2.putText(frame, label, (x1, y - 6), FONT, 0.6, WHITE, 2)


def draw_overlay(frame, landmarks, gesture, score, mapped, gesture_filter,
                 swipe_detector, t, last_action, last_action_t) -> None:
    h, w = frame.shape[:2]
    hand = landmarks is not None
    wrist = landmarks[0] if hand else None
    in_zone = hand and ZONE[0] <= wrist.x <= ZONE[2] and ZONE[1] <= wrist.y <= ZONE[3]

    # Strefa robocza
    cv2.rectangle(frame, _px(frame, ZONE[0], ZONE[1]), _px(frame, ZONE[2], ZONE[3]),
                  WHITE if in_zone else ORANGE, 1)

    # Punkty dloni i slad machniecia
    if hand:
        for lm in landmarks:
            cv2.circle(frame, _px(frame, lm.x, lm.y), 3, GREEN, -1)
        trail = [_px(frame, x, y) for x, y in swipe_detector.trail]
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, BLUE, 3)

    # Komunikat: co uzytkownik ma teraz zrobic
    if not hand:
        msg, color = "Pokaz dlon do kamery", GRAY
    elif not in_zone:
        msg, color = "Przesun dlon do ramki", ORANGE
    elif swipe_detector.is_cooling_down(t):
        msg, color = "Machniecie wykonane", BLUE
    elif gesture in mapped and gesture == gesture_filter.latched:
        msg, color = "Opusc reke, aby powtorzyc", BLUE
    else:
        msg, color = f"Gotowe: {gesture or '-'} {score:.2f}", GREEN
    cv2.putText(frame, msg, (10, 35), FONT, 0.8, color, 2)
    cv2.putText(frame, f"Ostatnia akcja: {last_action}", (10, 65), FONT, 0.6, WHITE, 2)

    # Paski postepu
    if gesture in mapped:
        _bar(frame, h - 75, gesture_filter.progress, GREEN, gesture)
    _bar(frame, h - 30, swipe_detector.progress, BLUE, "Machniecie")

    # Ramka statusu wokol calego obrazu
    flash = last_action_t is not None and t - last_action_t < FLASH_S
    if flash:
        border = BLUE
    elif in_zone:
        border = GREEN
    elif hand:
        border = ORANGE
    else:
        border = GRAY
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 10)