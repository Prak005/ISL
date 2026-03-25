import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x


DATASET_PATH = "dataset"
AUGMENT      = True
N_KP         = 63       


mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6,
)


def label_sort_key(name):
    if name.isdigit():
        return (0, int(name), "")  
    return (1, 0, name.upper())     


def normalize_hand(landmarks):
    pts  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()


def extract_from_image(image_bgr):
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    vecs = [normalize_hand(h) for h in results.multi_hand_landmarks]
    if len(vecs) == 1:
        vecs.append(np.zeros(N_KP))
    return np.concatenate(vecs[:2])


def augment(kp):
    variants = [kp.copy()]

    flipped = kp.copy()
    for start in (0, N_KP):
        for i in range(start, start + N_KP, 3):
            flipped[i] *= -1
    variants.append(flipped)

    variants.append(kp + np.random.normal(0, 0.005, kp.shape))
    variants.append(kp * np.random.uniform(0.9, 1.1))

    return variants


data, labels = [], []

label_list = sorted(
    [d for d in os.listdir(DATASET_PATH)
     if os.path.isdir(os.path.join(DATASET_PATH, d))],
    key=label_sort_key
)
label_map = {name: idx for idx, name in enumerate(label_list)}

print(f"Found {len(label_list)} classes:")
print(f"  {label_list}\n")
print("Label map (name → index):")
for name, idx in label_map.items():
    print(f"  {name:4s} → {idx}")
print()

for label in label_list:
    folder = os.path.join(DATASET_PATH, label)
    images = [f for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    count = 0
    for img_name in tqdm(images, desc=f"  {label}", leave=False):
        img = cv2.imread(os.path.join(folder, img_name))
        if img is None:
            continue
        kp = extract_from_image(img)
        if kp is None:
            continue
        for v in (augment(kp) if AUGMENT else [kp]):
            data.append(v)
            labels.append(label_map[label])
            count += 1

    print(f"{label:4s} → {count} samples")

X = np.array(data,   dtype=np.float32)
y = np.array(labels, dtype=np.int32)

with open("X.pkl",         "wb") as f: pickle.dump(X, f)
with open("y.pkl",         "wb") as f: pickle.dump(y, f)
with open("label_map.pkl", "wb") as f: pickle.dump(label_map, f)

print(f"\n{len(X)} samples  |  {X.shape[1]} features  |  {len(label_map)} classes")