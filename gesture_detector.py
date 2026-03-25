"""
Controls:
  SPACE     — commit word to sentence
  BACKSPACE — delete last letter
  C         — clear everything
  D         — toggle debug mode
  ESC       — quit

Auto: no hand for 2 seconds commits current word automatically
"""

import os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque


CONFIDENCE_THRESH  = 0.45
BUFFER_SIZE        = 6
HOLD_FRAMES        = 25
COOLDOWN_SECS      = 1.0
FRAME_SKIP         = 2
CAM_W, CAM_H       = 640, 480
AUTO_SPACE_SECS    = 2.0

EMOTION_MODEL_PATH  = "fer2013_mini_XCEPTION.119-0.65.hdf5"
EMOTION_LABELS_PATH = "labels_emotion.json"
ALLOWED_EMOTIONS    = {"Angry", "Happy", "Sad", "Surprise", "Neutral"}

N_KP = 63

def normalize_hand(landmarks):
    pts   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    pts  -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale
    return pts.flatten()

def extract_keypoints(hand_landmarks_list):
    vecs = [normalize_hand(h) for h in hand_landmarks_list]
    if len(vecs) == 1:
        vecs.append(np.zeros(N_KP))
    return np.concatenate(vecs[:2])


class EmotionDetector:

    def __init__(self):
        self.available    = False
        self.model        = None
        self.labels       = {}
        self.face_cascade = None

        try:
            from tensorflow.keras.models import load_model as keras_load

            if not os.path.exists(EMOTION_MODEL_PATH):
                print(f"Emotion model not found: {EMOTION_MODEL_PATH}")
                return
            if not os.path.exists(EMOTION_LABELS_PATH):
                print(f"Emotion labels not found: {EMOTION_LABELS_PATH}")
                return

            self.model = keras_load(EMOTION_MODEL_PATH, compile=False)

            with open(EMOTION_LABELS_PATH, "r") as f:
                self.labels = json.load(f)

            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            self.available = True
            print("Emotion detection loaded")

        except Exception as e:
            print(f"Emotion detection disabled: {e}")

    def predict(self, frame_gray):
        """Returns (emotion_str, x, y, w, h) or (None, 0,0,0,0) if no face."""
        if not self.available:
            return None, 0, 0, 0, 0

        faces = self.face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.3, minNeighbors=5
        )
        if len(faces) == 0:
            return None, 0, 0, 0, 0


        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        face = frame_gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        pred  = self.model.predict(face, verbose=0)
        idx   = int(np.argmax(pred))
        label = self.labels.get(str(idx), "Neutral")

        if label not in ALLOWED_EMOTIONS:
            label = "Neutral"

        return label, x, y, w, h


class ISLDetector:

    def __init__(self, model_path="sign_model.pkl"):
        
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            self.model        = bundle["model"]
            self.idx_to_label = bundle["idx_to_label"]
        else:
            self.model = bundle
            try:
                lm = pickle.load(open("label_map.pkl", "rb"))
                self.idx_to_label = {v: k for k, v in lm.items()}
            except FileNotFoundError:
                self.idx_to_label = {}

        
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_draw  = mp.solutions.drawing_utils
        self.dot_spec = self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=4)
        self.lin_spec = self.mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1)

        
        self.emotion_det     = EmotionDetector()
        self.current_emotion = "Neutral"

        
        self.buffer          = deque(maxlen=BUFFER_SIZE)
        self.sign_hold_count = 0
        self.last_added_time = 0.0
        self.last_added_sign = ""
        self.current_word    = []
        self.sentence        = []
        self.debug           = False
        self.top_preds       = []



    def predict(self, keypoints):
        kp = keypoints.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba     = self.model.predict_proba(kp)[0]
            top_idx   = np.argsort(proba)[::-1][:3]
            top3      = [(self.idx_to_label.get(int(i), str(i)), float(proba[i])) for i in top_idx]
            idx, conf = int(top_idx[0]), float(proba[top_idx[0]])
        else:
            idx  = int(self.model.predict(kp)[0])
            conf = 1.0
            top3 = [(self.idx_to_label.get(idx, str(idx)), 1.0)]

        if conf < CONFIDENCE_THRESH:
            return None, conf, top3
        return self.idx_to_label.get(idx, str(idx)), conf, top3


    def _confirm_sign(self, sign):
        now = time.time()
        if (now - self.last_added_time) > COOLDOWN_SECS or sign != self.last_added_sign:
            self.current_word.append(sign)
            self.last_added_time = now
            self.last_added_sign = sign
            self.sign_hold_count = 0

    def add_space(self):
        if self.current_word:
            self.sentence.append("".join(self.current_word))
            self.current_word = []

    def backspace(self):
        if self.current_word:
            self.current_word.pop()
        elif self.sentence:
            self.current_word = list(self.sentence.pop())

    def clear_all(self):
        self.current_word = []
        self.sentence     = []
        self.buffer.clear()

    def get_sentence_str(self):
        words = list(self.sentence)
        if self.current_word:
            words.append("".join(self.current_word))
        return " ".join(words)



    def _draw_hud(self, frame, sign, conf, no_hand_since):
        h, w = frame.shape[:2]

        if self.emotion_det.available:
            cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.line(frame, (0, 36), (w, 36), (50, 50, 50), 1)
            cv2.putText(frame, self.current_emotion,
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 255), 2, cv2.LINE_AA)

        
        cv2.rectangle(frame, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.line(frame, (0, h - 100), (w, h - 100), (50, 50, 50), 1)

        
        sign_txt = sign if sign else "..."
        sign_col = (0, 255, 120) if sign else (80, 80, 80)
        cv2.putText(frame, f"{sign_txt}  {int(conf*100)}%",
                    (10, h - 68), cv2.FONT_HERSHEY_DUPLEX, 1.2, sign_col, 2, cv2.LINE_AA)

        
        if no_hand_since is not None and self.current_word:
            remaining = max(0.0, AUTO_SPACE_SECS - (time.time() - no_hand_since))
            fill      = int((1.0 - remaining / AUTO_SPACE_SECS) * (w - 20))
            cv2.rectangle(frame, (10, h - 54), (w - 10, h - 46), (45, 45, 45), -1)
            if fill > 0:
                cv2.rectangle(frame, (10, h - 54), (10 + fill, h - 46), (0, 140, 255), -1)
            cv2.putText(frame, f"auto-space in {remaining:.1f}s",
                        (w - 178, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
        else:
            fill = int((self.sign_hold_count / HOLD_FRAMES) * (w - 20))
            cv2.rectangle(frame, (10, h - 54), (w - 10, h - 46), (45, 45, 45), -1)
            if fill > 0:
                cv2.rectangle(frame, (10, h - 54), (10 + fill, h - 46), (0, 210, 140), -1)

        
        word_str = "".join(self.current_word) if self.current_word else "_"
        cv2.putText(frame, f"Word: {word_str}",
                    (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 210, 255), 1, cv2.LINE_AA)

        
        sent_str = " ".join(self.sentence) if self.sentence else "_"
        if len(sent_str) > 55:
            sent_str = "..." + sent_str[-52:]
        cv2.putText(frame, f"Sentence: {sent_str}",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 220), 1, cv2.LINE_AA)

        
        if self.debug and self.top_preds:
            top_offset = 46 if self.emotion_det.available else 10
            for i, (lbl, c) in enumerate(self.top_preds):
                y_pos = top_offset + i * 28
                bw    = int(c * 180)
                col   = (0, 200, 100) if i == 0 else (0, 120, 70)
                cv2.rectangle(frame, (10, y_pos - 16), (10 + bw, y_pos), col, -1)
                cv2.putText(frame, f"{lbl} {c*100:.0f}%",
                            (14, y_pos - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)


    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("ISL Translator ready.")
        print("  SPACE=word  BACKSPACE=delete  C=clear  D=debug  ESC=quit")
        print(f"  Auto-space after {AUTO_SPACE_SECS}s with no hand\n")

        frame_idx     = 0
        last_sign     = ""
        last_conf     = 0.0
        last_result   = None
        no_hand_since = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame      = cv2.flip(frame, 1)
            frame_idx += 1

            
            if self.emotion_det.available and frame_idx % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                emotion, fx, fy, fw, fh = self.emotion_det.predict(gray)
                if emotion:
                    self.current_emotion = emotion
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (200, 200, 255), 1)

            
            if frame_idx % FRAME_SKIP == 0:
                rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_result = self.hands.process(rgb)

            results    = last_result
            sign, conf = None, 0.0

            if results and results.multi_hand_landmarks:
                no_hand_since = None

                for hl in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hl, self.mp_hands.HAND_CONNECTIONS,
                        self.dot_spec, self.lin_spec,
                    )

                kp               = extract_keypoints(results.multi_hand_landmarks)
                sign, conf, top3 = self.predict(kp)
                self.top_preds   = top3

                if sign:
                    self.buffer.append(sign)

                smoothed = max(set(self.buffer), key=self.buffer.count) if self.buffer else None

                if smoothed and smoothed == last_sign:
                    self.sign_hold_count += 1
                    if self.sign_hold_count >= HOLD_FRAMES:
                        self._confirm_sign(smoothed)
                else:
                    self.sign_hold_count = 0
                    if smoothed != last_sign:
                        self.buffer.clear()

                last_sign = smoothed or ""
                last_conf = conf

            else:
                self.top_preds       = []
                self.sign_hold_count = max(0, self.sign_hold_count - 1)
                last_sign, last_conf = "", 0.0

                if self.current_word:
                    if no_hand_since is None:
                        no_hand_since = time.time()
                    elif (time.time() - no_hand_since) >= AUTO_SPACE_SECS:
                        self.add_space()
                        no_hand_since = None

            self._draw_hud(frame, last_sign, last_conf, no_hand_since)
            cv2.imshow("ISL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == 27:                   break
            elif key == 32:                   self.add_space()
            elif key in (8, 127):             self.backspace()
            elif key in (ord("c"),ord("C")):  self.clear_all()
            elif key in (ord("d"),ord("D")):  self.debug = not self.debug

        cap.release()
        cv2.destroyAllWindows()
        print("Final sentence:", self.get_sentence_str())


if __name__ == "__main__":
    detector = ISLDetector("sign_model.pkl")
    detector.run()
