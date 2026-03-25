import pickle
import time
import numpy as np
from collections import Counter

from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix

print("Loading data...", end=" ", flush=True)
X         = pickle.load(open("X.pkl",         "rb"))
y         = pickle.load(open("y.pkl",         "rb"))
label_map = pickle.load(open("label_map.pkl", "rb"))

idx_to_label = {v: k for k, v in label_map.items()}
y = np.array(y)

print("done.\n")
print(f"  Samples  : {len(X)}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {len(set(y))}")
print(f"  Balance  : {dict(Counter(y))}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )),
])

print("Training...", end=" ", flush=True)
t0 = time.time()
model.fit(X_train, y_train)
print(f"done in {time.time()-t0:.1f}s\n")

y_pred   = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_acc*100:.2f}%\n")

present_labels = sorted(set(y_test))
target_names   = [idx_to_label[i] for i in present_labels]

print("Classification Report:")
print(classification_report(y_test, y_pred,
                             labels=present_labels,
                             target_names=target_names))

if len(present_labels) <= 36:
    print("Confusion Matrix (rows=true, cols=pred):")
    cm     = confusion_matrix(y_test, y_pred, labels=present_labels)
    header = "".join(f"{n:>5}" for n in target_names)
    print(f"{'':6}{header}")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:6}" + "".join(f"{v:>5}" for v in row))


bundle = {
    "model"       : model,
    "label_map"   : label_map,
    "idx_to_label": idx_to_label,
    "feature_dim" : int(X.shape[1]),
    "accuracy"    : float(test_acc),
}

with open("sign_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print(f"\nSaved sign_model.pkl  (test acc: {test_acc*100:.2f}%)")
