import os
import librosa
import numpy as np
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


test_dir = "./data/Standardized_Data/testing/"  
model_path = "./models/xgb_model_1.pkl"

def compute_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = np.hstack((
        np.mean(rms), np.mean(spec_cent), np.mean(spec_bw),
        np.mean(rolloff), np.mean(zcr), np.mean(mfcc, axis=1)
    ))
    return features

X_test, y_test = [], []
classes = {"COPD_Segments": 0, "Healthy_Segments": 1}
for label, class_num in classes.items():
    folder = os.path.join(test_dir, label)
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            features = compute_features(file_path)
            X_test.append(features)
            y_test.append(class_num)

X_test = np.array(X_test)
y_test = np.array(y_test)

model = joblib.load(model_path)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["COPD", "Healthy"]))


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["COPD", "Healthy"], yticklabels=["COPD", "Healthy"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()