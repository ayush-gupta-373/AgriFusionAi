import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 1. Setup Directories
DATA_PATH = r"c:\AgriFusionAi\data\fertilizer_recommendation.csv"
MODELS_DIR = r"c:\AgriFusionAi\models"
PAPER_DIR = r"c:\AgriFusionAi\research_paper"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

print("1. Loading Fertilizer Recommendation Dataset (10,000 records)...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop(columns=["Recommended_Fertilizer"])
y = df["Recommended_Fertilizer"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"   Fertilizer Classes ({len(class_names)}): {', '.join(class_names)}")

# Identify Column Types
cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Previous_Crop', 'Region']
num_cols = [c for c in X.columns if c not in cat_cols]

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 2. Define Pipelines with Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

models = {
    "Random Forest (Proposed)": Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ('prep', preprocessor),
        ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "Decision Tree (Baseline)": Pipeline([
        ('prep', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    "Multi-Layer Perceptron (ANN)": Pipeline([
        ('prep', preprocessor),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
    ])
}

print("\n2. Benchmarking Algorithms with 5-Fold Stratified Cross-Validation...")
print("-" * 85)

results = []
best_f1 = 0.0
best_model_name = ""
best_model_obj = None
best_preds = None

for name, pipeline in models.items():
    # Measure Training Time
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - start_train, 3)

    # Measure Inference Latency
    start_inf = time.time()
    y_pred = pipeline.predict(X_test)
    latency_per_sample_ms = round(((time.time() - start_inf) / len(X_test)) * 1000, 4)

    # Calculate Metrics
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100, 2)

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    cv_mean = round(cv_scores.mean() * 100, 2)
    cv_std = round(cv_scores.std() * 100, 2)

    results.append({
        "Model Architecture": name,
        "Accuracy (%)": acc,
        "Macro Precision (%)": prec,
        "Macro Recall (%)": rec,
        "Macro F1-Score (%)": f1,
        "5-Fold CV Score (%)": f"{cv_mean} +/- {cv_std}",
        "Inference Latency (ms)": latency_per_sample_ms,
        "Train Time (s)": train_time
    })

    print(f" -> {name:<30} | F1: {f1:.2f}% | Acc: {acc:.2f}% | Latency: {latency_per_sample_ms:.4f} ms")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = pipeline
        best_preds = y_pred

# 3. Create DataFrame and Print Results
benchmark_df = pd.DataFrame(results)
print("\n" + "=" * 85)
print("FINAL FERTILIZER BENCHMARK TABLE FOR RESEARCH PAPER:")
print("=" * 85)
print(benchmark_df.to_string(index=False))

# Save Benchmark Table to CSV
csv_out = os.path.join(PAPER_DIR, "fertilizer_model_comparison_benchmark.csv")
benchmark_df.to_csv(csv_out, index=False)
print(f"\n Saved Benchmark Table: {csv_out}")

# 4. Generate & Save High-Resolution Confusion Matrix (300 DPI)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix: {best_model_name} (Macro F1: {best_f1}%)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Fertilizer", fontsize=11)
plt.ylabel("True Fertilizer", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_out = os.path.join(PAPER_DIR, "confusion_matrix_fertilizer_prediction.png")
plt.savefig(cm_out, dpi=300)
print(f" Saved Publication Plot (300 DPI): {cm_out}")

# 5. Save the Winning Model Package as fertilizerprediction.pkl
package = {
    "pipeline": best_model_obj,
    "label_encoder": le,
    "class_names": list(class_names),
    "features": list(X.columns),
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "model_name": best_model_name,
    "f1_score": best_f1
}

save_path = os.path.join(MODELS_DIR, "fertilizerprediction.pkl")
with open(save_path, "wb") as f:
    pickle.dump(package, f)

print(f"\n SUCCESS: Winning Model ({best_model_name}) saved successfully at:")
print(f" {save_path}")
