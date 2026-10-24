"""
Handwritten Digit Recognition using Machine Learning and Neural Networks
Dataset: scikit-learn load_digits (1,797 samples, 8x8 grayscale, labels 0-9)

Run:
    python main.py
Optional:
    python main.py --no-sensitivity
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, learning_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def build_models():
    """Return the six classifiers described in the internship report."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=10, gamma="scale"))
        ]),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", max_iter=500, random_state=RANDOM_STATE
            ))
        ]),
    }


def save_dataset_visuals(images, labels):
    # Sample digits
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for digit, ax in enumerate(axes.ravel()):
        idx = np.where(labels == digit)[0][0]
        ax.imshow(images[idx], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"Label: {digit}")
        ax.axis("off")
    fig.suptitle("Sample Images from the Digits Dataset")
    fig.tight_layout()
    fig.savefig(OUT / "sample_digits.png", dpi=160)
    plt.close(fig)

    # Class distribution
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set(title="Number of Samples per Digit Class", xlabel="Digit", ylabel="Samples")
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=160)
    plt.close(fig)

    # PCA projection
    reduced = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        images.reshape(len(images), -1)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=14, alpha=.75)
    ax.set(title="PCA Projection of the Digits Dataset", xlabel="Principal Component 1",
           ylabel="Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Digit")
    fig.tight_layout()
    fig.savefig(OUT / "pca_projection.png", dpi=160)
    plt.close(fig)


def evaluate_models(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in build_models().items():
        print(f"Training: {name}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        rows.append({
            "Model": name,
            "CV accuracy mean (%)": round(cv_scores.mean() * 100, 2),
            "CV accuracy std (%)": round(cv_scores.std() * 100, 2),
            "Test accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
            "Weighted precision": round(precision, 4),
            "Weighted recall": round(recall, 4),
            "Weighted F1": round(f1, 4),
        })
        fitted[name] = (model, predictions)
    results = pd.DataFrame(rows).sort_values("Test accuracy (%)", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nModel comparison:\n")
    print(results.to_string(index=False))

    # Accuracy comparison chart
    chart = results.sort_values("Test accuracy (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(chart["Model"], chart["Test accuracy (%)"])
    ax.set(xlabel="Test accuracy (%)", title="Test Accuracy of Six Classifiers", xlim=(0, 100))
    for i, val in enumerate(chart["Test accuracy (%)"]):
        ax.text(val + .15, i, f"{val:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "model_accuracy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    best_name = results.iloc[0]["Model"]
    best_model, best_predictions = fitted[best_name]
    joblib.dump(best_model, MODELS / "best_digit_classifier.joblib")
    with open(OUT / "best_model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model by held-out test accuracy: {best_name}\n\n")
        f.write(classification_report(y_test, best_predictions, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, best_predictions, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(OUT / "best_model_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Save example prediction grid
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(len(X_test), size=min(20, len(X_test)), replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(10, 7))
    for ax, idx in zip(axes.ravel(), chosen):
        ax.imshow(X_test[idx].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        pred = best_model.predict(X_test[idx].reshape(1, -1))[0]
        ax.set_title(f"True {y_test[idx]} / Pred {pred}", color=("green" if pred == y_test[idx] else "red"))
        ax.axis("off")
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Held-out Test Predictions (green = correct, red = incorrect)")
    fig.tight_layout()
    fig.savefig(OUT / "prediction_examples.png", dpi=160)
    plt.close(fig)

    return results, best_name, best_model


def sensitivity_experiments(X_train, X_test, y_train, y_test):
    """Small, reproducible hyperparameter checks inspired by the report."""
    rows = []

    # SVM: raw vs standardized pixels and different kernels/C
    svm_variants = [
        ("SVM RBF, raw pixels, C=10", SVC(kernel="rbf", C=10, gamma="scale")),
        ("SVM RBF, scaled, C=1", Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1))])),
        ("SVM RBF, scaled, C=10", Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=10))])),
        ("SVM linear, scaled, C=1", Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="linear", C=1))])),
    ]
    for name, model in svm_variants:
        model.fit(X_train, y_train)
        rows.append({"Experiment": name, "Test accuracy (%)": round(accuracy_score(y_test, model.predict(X_test))*100, 2)})

    # k-NN across k
    for k in [1, 3, 5, 7, 15]:
        model = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=k))])
        model.fit(X_train, y_train)
        rows.append({"Experiment": f"k-NN scaled, k={k}", "Test accuracy (%)": round(accuracy_score(y_test, model.predict(X_test))*100, 2)})

    # Random forest sizes
    for trees in [10, 50, 100, 200]:
        model = RandomForestClassifier(n_estimators=trees, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        rows.append({"Experiment": f"Random Forest, trees={trees}", "Test accuracy (%)": round(accuracy_score(y_test, model.predict(X_test))*100, 2)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity_experiments.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    chart = df.sort_values("Test accuracy (%)")
    ax.barh(chart["Experiment"], chart["Test accuracy (%)"])
    ax.set(xlabel="Test accuracy (%)", title="Sensitivity Experiments", xlim=(0, 100))
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_analysis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("\nSensitivity experiments saved to outputs/sensitivity_experiments.csv")


def main():
    parser = argparse.ArgumentParser(description="Train and compare handwritten digit classifiers.")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip additional parameter experiments.")
    args = parser.parse_args()

    digits = load_digits()
    images = digits.images
    X = digits.data.astype(np.float64)
    y = digits.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, classes {sorted(np.unique(y))}")
    print(f"Pixel intensity range: {X.min()} to {X.max()}")

    save_dataset_visuals(images, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    results, best_name, best_model = evaluate_models(X_train, X_test, y_train, y_test)

    if not args.no_sensitivity:
        sensitivity_experiments(X_train, X_test, y_train, y_test)

    metadata = {
        "dataset": "sklearn.datasets.load_digits",
        "samples": int(len(X)),
        "features_per_image": int(X.shape[1]),
        "classes": [int(x) for x in np.unique(y)],
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "best_model_by_test_accuracy": best_name,
        "best_test_accuracy_percent": float(results.iloc[0]["Test accuracy (%)"]),
    }
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nBest model: {best_name}")
    print(f"Test accuracy: {results.iloc[0]['Test accuracy (%)']}%")
    print(f"Saved model: {MODELS / 'best_digit_classifier.joblib'}")
    print(f"Outputs folder: {OUT}")


if __name__ == "__main__":
    main()
