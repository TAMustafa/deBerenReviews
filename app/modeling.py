import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC


def train_sentiment_model(X_train, y_train):
    """Train and select between LogisticRegression and LinearSVC by macro F1."""
    _, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts)) if len(counts) else 0
    n_splits = int(np.clip(min_count, 2, 3))

    candidates = []
    for C in [0.1, 0.5, 1.0, 3.0, 10.0]:
        candidates.append((f"logreg_C={C}", LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs", C=C)))
    for C in [0.5, 1.0, 3.0, 10.0]:
        candidates.append((f"linsvc_C={C}", LinearSVC(C=C, class_weight="balanced")))

    best_name = None
    best_model = None
    best_score = -1.0
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for name, clf in candidates:
            try:
                scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
                score = float(np.nanmean(scores))
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_name = name
                    best_model = clf
            except Exception:
                continue
    if best_model is None:
        best_name, best_model = candidates[0]

    best_model.fit(X_train, y_train)
    return best_model, best_name
