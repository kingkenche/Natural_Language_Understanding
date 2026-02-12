"""
train.py — Train 3 classifiers × 3 feature sets and save results.
"""

import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# allow importing siblings
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data          # noqa: E402
from features import get_vectorizers       # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial NB": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
    }


def run_experiments():
    X_train, X_test, y_train, y_test, _ = load_data()

    vectorizers = get_vectorizers()
    classifiers = get_classifiers()

    all_results = []

    for feat_name, vec in vectorizers.items():
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        for clf_name, clf in classifiers.items():
            print(f"\n{'='*60}")
            print(f"  Feature: {feat_name}  |  Classifier: {clf_name}")
            print(f"{'='*60}")

            # 5-fold CV
            cv_scores = cross_val_score(clf, X_tr, y_train, cv=5, scoring="accuracy")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Train on full train set, evaluate on test
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, target_names=["Politics", "Sport"], output_dict=True
            )
            cm = confusion_matrix(y_test, y_pred)

            print(f"  Test Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, target_names=["Politics", "Sport"]))

            result = {
                "feature": feat_name,
                "classifier": clf_name,
                "cv_accuracy_mean": round(cv_scores.mean(), 4),
                "cv_accuracy_std": round(cv_scores.std(), 4),
                "test_accuracy": round(acc, 4),
                "precision_politics": round(report["Politics"]["precision"], 4),
                "recall_politics": round(report["Politics"]["recall"], 4),
                "f1_politics": round(report["Politics"]["f1-score"], 4),
                "precision_sport": round(report["Sport"]["precision"], 4),
                "recall_sport": round(report["Sport"]["recall"], 4),
                "f1_sport": round(report["Sport"]["f1-score"], 4),
                "macro_f1": round(report["macro avg"]["f1-score"], 4),
                "confusion_matrix": cm.tolist(),
            }
            all_results.append(result)

            # Re-initialise classifier for next experiment
            classifiers[clf_name] = type(clf)(**clf.get_params())

    # Save results
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] All results saved → {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiments()
