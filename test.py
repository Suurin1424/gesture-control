import time
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = Path(__file__).with_name("models") / "gesture_recognizer.task"
WINDOW = "MediaPipe"


def main() -> None:
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nie udalo sie otworzyc kamery (indeks 0).")

    last_ts = -1
    with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Brak klatki z kamery.")
                    break
                # Odbicie PRZED rozpoznaniem: punkty beda zgodne z podgladem.
                frame = cv2.flip(frame, 1)

                # OpenCV daje BGR, MediaPipe oczekuje RGB.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Tryb VIDEO wymaga rosnacych znacznikow czasu w ms.
                ts = max(int(time.perf_counter() * 1000), last_ts + 1)
                last_ts = ts
                result = recognizer.recognize_for_video(mp_image, ts)

                h, w = frame.shape[:2]
                if result.hand_landmarks:
                    # Wspolrzedne sa znormalizowane do [0, 1], przeliczamy na piksele.
                    for lm in result.hand_landmarks[0]:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                if result.gestures:
                    top = result.gestures[0][0]
                    cv2.putText(frame, f"{top.category_name} {top.score:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()