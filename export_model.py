#!/usr/bin/env python3
"""Export the trained scikit-learn RandomForest for use in the browser demo.

The web app reproduces the same feature extraction and prediction pipeline as
``gesture_recognition.py``.  This script flattens every decision tree into a
compact JSON structure and writes a handful of dataset rows together with the
model's expected output so the JavaScript inference can be verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gesture_model.pkl"
DATASET_PATH = ROOT / "dataset" / "gestures.csv"
OUTPUT_DIR = ROOT / "web" / "model"


def export_tree(estimator) -> dict:
    """Convert one ``DecisionTreeClassifier`` into flat arrays."""
    tree = estimator.tree_

    probabilities = []
    for node in range(tree.node_count):
        if tree.children_left[node] == -1:
            counts = tree.value[node, 0].astype(np.float64)
            total = counts.sum()
            values = counts / total if total else np.zeros_like(counts)
            probabilities.append([round(float(value), 8) for value in values])
        else:
            probabilities.append(None)

    return {
        "f": tree.feature.astype(int).tolist(),
        "t": [round(float(value), 7) for value in tree.threshold],
        "l": tree.children_left.astype(int).tolist(),
        "r": tree.children_right.astype(int).tolist(),
        "p": probabilities,
    }


def predict(model, features: np.ndarray) -> tuple[int, np.ndarray]:
    """Mirror the JavaScript inference: average the leaf distributions."""
    sums = np.zeros(len(model.classes_), dtype=np.float64)
    for estimator in model.estimators_:
        tree = estimator.tree_
        node = 0
        while tree.children_left[node] != -1:
            feature = tree.feature[node]
            node = (
                tree.children_left[node]
                if features[feature] <= tree.threshold[node]
                else tree.children_right[node]
            )
        counts = tree.value[node, 0].astype(np.float64)
        total = counts.sum()
        if total:
            sums += counts / total
    sums /= len(model.estimators_)
    return int(np.argmax(sums)), sums


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "classes": [str(label) for label in model.classes_],
        "n_features": int(model.n_features_in_),
        "n_estimators": int(len(model.estimators_)),
        "trees": [export_tree(estimator) for estimator in model.estimators_],
    }

    model_file = OUTPUT_DIR / "gesture_model.json"
    model_file.write_text(
        json.dumps(model_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Wrote {model_file} ({model_file.stat().st_size / 1024:.1f} KiB)")
    print(f"Classes: {', '.join(model_payload['classes'])}")

    if not DATASET_PATH.exists():
        print(f"Dataset not found, skipping test vectors: {DATASET_PATH}")
        return

    import pandas as pd

    frame = pd.read_csv(DATASET_PATH)
    feature_columns = [f"feature_{index}" for index in range(model.n_features_in_)]

    # Pick a few rows per gesture so the test covers every class.
    sample = (
        frame.groupby("label", group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )

    cases = []
    for _, row in sample.iterrows():
        features = row[feature_columns].to_numpy(dtype=np.float64)
        predicted, probabilities = predict(model, features)
        cases.append(
            {
                "label": str(row["label"]),
                "features": [float(value) for value in features],
                "expected": str(model.classes_[predicted]),
                "probabilities": [round(float(value), 8) for value in probabilities],
            }
        )

    vectors_file = OUTPUT_DIR / "test_vectors.json"
    vectors_file.write_text(
        json.dumps({"cases": cases}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {vectors_file} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
