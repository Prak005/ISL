"""
Controls:
  SPACE  — skip countdown, start capturing immediately
  ENTER  — skip current class
  B      — go back to previous class
  R      — redo current class also deletes this session's captures
  ESC    — quit
"""

import cv2
import mediapipe as mp
import os
import time


DATASET_PATH       = "dataset"
CAPTURES_PER_CLASS = 200        
COUNTDOWN_SECS     = 3          
CAPTURE_DELAY      = 0.05       
HAND_REQUIRED      = True       
CAM_W, CAM_H       = 640, 480


NEW_FILE_PREFIX = "my_"

CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]


mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
)
mp_draw  = mp.solutions.drawing_utils
dot_spec = mp_draw.DrawingSpec(color=(0, 255, 180), thickness=-1, circle_radius=3)
lin_spec = mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)




def get_next_new_index(folder):
    existing = []
    for f in os.listdir(folder):
        if f.startswith(NEW_FILE_PREFIX) and f.endswith(".jpg"):
            try:
                idx = int(f[len(NEW_FILE_PREFIX):-4])
                existing.append(idx)
            except ValueError:
                pass
    return max(existing) + 1 if existing else 0


def count_existing_originals(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if not f.startswith(NEW_FILE_PREFIX)
        and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )


def load_reference(label):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if not f.startswith(NEW_FILE_PREFIX) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                return cv2.resize(img, (200, 200))
    return None


def draw_hud(frame, label, captured, total, countdown, hand_detected, ref_img, orig_count):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"Class: {label}", (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 180), 2)
    cv2.putText(frame, f"New: {captured}/{total}", (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    bar_w = int((captured / max(total, 1)) * (w - 20))
    cv2.rectangle(frame, (10, 50), (w - 10, 62), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 62), (0, 200, 100), -1)

    cv2.putText(frame, f"Original dataset: {orig_count} imgs (untouched)",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 80), 1)

    hand_col = (0, 255, 120) if hand_detected else (0, 80, 255)
    cv2.putText(frame, "Hand: YES" if hand_detected else "Hand: NO",
                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_col, 2)

    if countdown > 0:
        txt, col = f"Starting in {countdown:.1f}s  (SPACE to skip)", (0, 180, 255)
    else:
        txt, col = "● CAPTURING", (0, 255, 80)
    cv2.putText(frame, txt, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    cv2.putText(frame, "[SPC] instant  [ENTER] next  [B] back  [R] redo session  [ESC] quit",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

    if ref_img is not None:
        rh, rw = ref_img.shape[:2]
        frame[65:65 + rh, w - rw - 10:w - 10] = ref_img
        cv2.putText(frame, "Dataset ref", (w - rw - 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    class_idx   = 0
    session_new = {}   

    print(f"   New images saved as: {NEW_FILE_PREFIX}<number>.jpg\n")

    while class_idx < len(CLASSES):
        label      = CLASSES[class_idx]
        folder     = os.path.join(DATASET_PATH, label)
        os.makedirs(folder, exist_ok=True)

        ref_img    = load_reference(label)
        orig_count = count_existing_originals(folder)
        captured   = 0
        next_idx   = get_next_new_index(folder)
        session_new[label] = []

        countdown_start = time.time()
        last_save       = 0.0
        capturing       = False

        print(f"[{class_idx+1}/{len(CLASSES)}] '{label}'  —  {orig_count} original images present")

        while captured < CAPTURES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb           = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results       = hands.process(rgb)
            hand_detected = results.multi_hand_landmarks is not None

            if hand_detected:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                           dot_spec, lin_spec)

            elapsed   = time.time() - countdown_start
            countdown = max(0.0, COUNTDOWN_SECS - elapsed)
            if countdown == 0:
                capturing = True

            now     = time.time()
            save_ok = (
                capturing
                and (now - last_save) >= CAPTURE_DELAY
                and (not HAND_REQUIRED or hand_detected)
            )
            if save_ok:
                filename = f"{NEW_FILE_PREFIX}{next_idx}.jpg"
                path     = os.path.join(folder, filename)
                cv2.imwrite(path, frame)
                session_new[label].append(path)
                next_idx += 1
                captured += 1
                last_save = now

            draw_hud(frame, label, captured, CAPTURES_PER_CLASS,
                     countdown, hand_detected, ref_img, orig_count)
            cv2.imshow("ISL Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:                           
                cap.release()
                cv2.destroyAllWindows()
                total = sum(len(v) for v in session_new.values())
                print(f"\nStopped early. {total} new images saved.")
                return

            elif key == 32:                         
                countdown_start = time.time() - COUNTDOWN_SECS

            elif key == 13:                         
                print(f"Skipped after {captured} new images.")
                break

            elif key in (ord("b"), ord("B")):       
                class_idx = max(0, class_idx - 2)
                break

            elif key in (ord("r"), ord("R")):       
                deleted = 0
                for p in session_new.get(label, []):
                    if os.path.exists(p):
                        os.remove(p)
                        deleted += 1
                print(f"Redo: deleted {deleted} session images for '{label}'")
                captured        = 0
                next_idx        = get_next_new_index(folder)
                session_new[label] = []
                countdown_start = time.time()
                capturing       = False
        else:
            print(f"{captured} new images saved for '{label}'")

        class_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    total = sum(len(v) for v in session_new.values())

if __name__ == "__main__":
    main()
