"""Logistic regression over SVI parameters for regime classification.

We generate a small labelled synthetic dataset of (a, b, rho, m, sigma) tuples
drawn from three regime prototypes (calm, stressed, crash) with jitter, fit
a multiclass logistic regression, then classify the current slice set by
majority vote across tenors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

REGIMES = ("calm", "stressed", "crash")


def _prototype(regime: str) -> dict[str, float]:
    if regime == "calm":
        return {"a": 0.010, "b": 0.040, "rho": -0.25, "m": 0.00, "sigma": 0.30}
    if regime == "stressed":
        return {"a": 0.030, "b": 0.110, "rho": -0.55, "m": -0.05, "sigma": 0.20}
    return {"a": 0.070, "b": 0.250, "rho": -0.85, "m": -0.15, "sigma": 0.12}


@dataclass
class RegimeClassifier:
    model: LogisticRegression
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    train_accuracy: float
    test_accuracy: float
    confusion: np.ndarray
    inferred_regime: str = "unknown"
    per_slice: list[str] = field(default_factory=list)
    per_slice_probs: list[list[float]] = field(default_factory=list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.scaler_mean) / np.maximum(self.scaler_std, 1e-9)
        return self.model.predict(xs)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.scaler_mean) / np.maximum(self.scaler_std, 1e-9)
        return self.model.predict_proba(xs)


def _generate_labelled(n_per_class: int = 160, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    jitter_std = {
        "a": 0.005,
        "b": 0.020,
        "rho": 0.08,
        "m": 0.05,
        "sigma": 0.06,
    }
    for cls, name in enumerate(REGIMES):
        proto = _prototype(name)
        for _ in range(n_per_class):
            sample = np.array(
                [
                    proto["a"] + rng.normal(0.0, jitter_std["a"]),
                    max(0.0, proto["b"] + rng.normal(0.0, jitter_std["b"])),
                    float(np.clip(proto["rho"] + rng.normal(0.0, jitter_std["rho"]), -0.98, 0.98)),
                    proto["m"] + rng.normal(0.0, jitter_std["m"]),
                    max(0.05, proto["sigma"] + rng.normal(0.0, jitter_std["sigma"])),
                ]
            )
            xs.append(sample)
            ys.append(cls)
    return np.stack(xs, axis=0), np.array(ys, dtype=int)


def train_regime_classifier(
    slice_params: list[dict[str, float]],
    seed: int = 11,
) -> RegimeClassifier:
    x, y = _generate_labelled(seed=seed)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    xs = (x - mean) / np.maximum(std, 1e-9)
    n_train = int(0.8 * xs.shape[0])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(xs.shape[0])
    tr, te = perm[:n_train], perm[n_train:]
    clf = LogisticRegression(max_iter=500)
    clf.fit(xs[tr], y[tr])
    tr_acc = float(clf.score(xs[tr], y[tr]))
    te_acc = float(clf.score(xs[te], y[te]))
    cm = confusion_matrix(y[te], clf.predict(xs[te]), labels=[0, 1, 2])

    per_slice: list[str] = []
    per_slice_probs: list[list[float]] = []
    if slice_params:
        xp = np.array(
            [
                [s["a"], s["b"], s["rho"], s["m"], s["sigma"]]
                for s in slice_params
            ]
        )
        xps = (xp - mean) / np.maximum(std, 1e-9)
        preds = clf.predict(xps)
        probs = clf.predict_proba(xps)
        per_slice = [REGIMES[int(p)] for p in preds]
        per_slice_probs = probs.tolist()
        # Majority vote across slices.
        counts = np.bincount(preds, minlength=3)
        inferred = REGIMES[int(np.argmax(counts))]
    else:
        inferred = "unknown"

    return RegimeClassifier(
        model=clf,
        scaler_mean=mean,
        scaler_std=std,
        train_accuracy=tr_acc,
        test_accuracy=te_acc,
        confusion=cm,
        inferred_regime=inferred,
        per_slice=per_slice,
        per_slice_probs=per_slice_probs,
    )
