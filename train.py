import os
import librosa
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

train_dir = "./data/Standardized_Data/training/"  
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

X_train, y_train = [], []
classes = {"COPD_Segments": 0, "Healthy_Segments": 1}
for label, class_num in classes.items():
    folder = os.path.join(train_dir, label)
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            features = compute_features(file_path)
            X_train.append(features)
            y_train.append(class_num)

X_train = np.array(X_train)
y_train = np.array(y_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(n_jobs=-1, random_state=42),
    param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

joblib.dump(best_model, model_path)
