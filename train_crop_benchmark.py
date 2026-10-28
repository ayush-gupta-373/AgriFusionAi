import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Setup Directories
DATA_PATH = r"c:\AgriFusionAi\data\Crop_recommendation.csv"
MODELS_DIR = r"c:\AgriFusionAi\models"
PAPER_DIR = r"c:\AgriFusionAi\research_paper"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

print("1. Loading Crop Recommendation Dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop(columns=["label"])
y = df["label"]

# Encode Crop Labels (22 crops)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"   Total Crops ({len(class_names)}): {', '.join(class_names[:8])}...")

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature Scaling for SVM and MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check optional XGBoost
models = {
    "Random Forest (Proposed)": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "Decision Tree (Baseline)": (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "Support Vector Machine (RBF)": (SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_test_scaled),
    "Multi-Layer Perceptron (ANN)": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), X_train_scaled, X_test_scaled)
}

try:
    from xgboost import XGBClassifier
    models["XGBoost Classifier"] = (XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss"), X_train, X_test)
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    models["LightGBM Classifier"] = (LGBMClassifier(random_state=42, verbose=-1), X_train, X_test)
except ImportError:
    pass

print("\n2. Benchmarking Algorithms across 5-Fold Stratified Cross-Validation...")
print("-" * 85)

results = []
best_f1 = 0.0
best_model_name = ""
best_model_obj = None
best_preds = None

for name, (clf, x_tr, x_te) in models.items():
    # Measure Training Time
    start_train = time.time()
    clf.fit(x_tr, y_train)
    train_time = round(time.time() - start_train, 3)

    # Measure Inference Latency
    start_inf = time.time()
    y_pred = clf.predict(x_te)
    latency_per_sample_ms = round(((time.time() - start_inf) / len(x_te)) * 1000, 4)

    # Metrics
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)

    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(clf, x_tr, y_train, cv=5)
    cv_mean = round(cv_scores.mean() * 100, 2)
    cv_std = round(cv_scores.std() * 100, 2)

    results.append({
        "Model Architecture": name,
        "Accuracy (%)": acc,
        "Macro Precision (%)": prec,
        "Macro Recall (%)": rec,
        "Macro F1-Score (%)": f1,
        "5-Fold CV Score (%)": f"{cv_mean} ± {cv_std}",
        "Inference Latency (ms)": latency_per_sample_ms,
        "Train Time (s)": train_time
    })

    print(f" -> {name:<30} | F1: {f1:.2f}% | Acc: {acc:.2f}% | Latency: {latency_per_sample_ms:.4f} ms")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = clf
        best_preds = y_pred

# 3. Create DataFrame and Print Results
benchmark_df = pd.DataFrame(results)
print("\n" + "=" * 85)
print("FINAL RESEARCH PAPER TABULAR BENCHMARK (Table II):")
print("=" * 85)
print(benchmark_df.to_string(index=False))

# Save Benchmark Table to CSV
csv_out = os.path.join(PAPER_DIR, "crop_model_comparison_benchmark.csv")
benchmark_df.to_csv(csv_out, index=False)
print(f"\n Saved Benchmark Table: {csv_out}")

# 4. Generate & Save High-Resolution Confusion Matrix (300 DPI)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(14, 11), dpi=300)
sns.heatmap(cm, annot=False, cmap="Greens", cbar=True)
plt.title(f"Confusion Matrix: {best_model_name} (Macro F1: {best_f1}%)", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Crop Class Index", fontsize=12)
plt.ylabel("True Crop Class Index", fontsize=12)
plt.tight_layout()

cm_out = os.path.join(PAPER_DIR, "confusion_matrix_crop_prediction.png")
plt.savefig(cm_out, dpi=300)
print(f" Saved Publication Plot (300 DPI): {cm_out}")

# 5. Save the Winning Model as cropprediction.pkl
# Package the model together with its label encoder so FastAPI can easily decode crop names
model_package = {
    "model": best_model_obj,
    "label_encoder": le,
    "class_names": class_names,
    "features": list(X.columns),
    "model_name": best_model_name,
    "f1_score": best_f1
}

save_path = os.path.join(MODELS_DIR, "cropprediction.pkl")
with open(save_path, "wb") as f:
    pickle.dump(model_package, f)

print(f"\n SUCCESS: Winning Model ({best_model_name}) saved as:")
print(f" 👉 {save_path}")
