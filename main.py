import json
import time
import cv2
import mediapipe as mp

from pathlib import Path
from actions import KeyboardActions
from gesture_filter import GestureFilter
from swipe import SwipeDetector
from overlay import draw_overlay

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "gesture_recognizer.task"
CONFIG_PATH = BASE_DIR / "config.json"
WINDOW = "Sterowanie gestami"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def create_recognizer():
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def top_gesture(result) -> tuple[str | None, float]:
    """Najbardziej prawdopodobny gest pierwszej wykrytej dloni."""
    if not result.gestures:
        return None, 0.0
    best = result.gestures[0][0]
    return best.category_name, best.score


def main() -> None:
    config = load_config()
    gesture_filter = GestureFilter(
        stable_frames=config["stable_frames"],
        release_frames=config["release_frames"],
        min_score=config["min_score"],
    )
    actions = KeyboardActions(config["mappings"])
    swipe_detector = SwipeDetector(**config["swipe"])

    cam = config["camera"]
    cap = cv2.VideoCapture(cam["index"])
    if not cap.isOpened():
        raise RuntimeError(f"Nie udalo sie otworzyc kamery (indeks {cam['index']}).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

    # Kamera moze nie obslugiwac zadanej rozdzielczosci i wybrac najblizsza - sprawdzamy, co dostalismy.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Rozdzielczosc kamery: {width}x{height}")

    last_ts = -1
    last_action = "-"
    last_action_t = None
    mapped = set(config["mappings"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)          # NORMAL pozwala zmienic rozmiar
    preview_size = (640, int(640 * height / width))  # (szerokosc, wysokosc) - kolejnosc dla cv2.resize
    cv2.resizeWindow(WINDOW, 480, int(480 * height / width))
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    with create_recognizer() as recognizer:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Brak klatki z kamery.")
                    break
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = max(int(time.perf_counter() * 1000), last_ts + 1)
                last_ts = ts
                t = ts / 1000
                result = recognizer.recognize_for_video(mp_image, ts)

                gesture, score = top_gesture(result)
                fired = gesture_filter.update(gesture, score)
                if fired is not None and actions.execute(fired):
                    last_action, last_action_t = fired, t
                    print(f"Akcja: {fired}")

                landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                swipe = swipe_detector.update(landmarks, gesture, t)
                if swipe is not None and actions.execute(swipe):
                    last_action, last_action_t = swipe, t
                    print(f"Akcja: {swipe}")

                preview = cv2.resize(frame, preview_size)
                draw_overlay(preview, landmarks, gesture, score, mapped, gesture_filter,
                             swipe_detector, t, last_action, last_action_t)

                cv2.imshow(WINDOW, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()