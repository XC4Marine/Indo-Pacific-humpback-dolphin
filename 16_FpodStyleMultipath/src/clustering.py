from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from .multipath_detection import MultipathFeature


@dataclass(frozen=True)
class ClusteringResult:
    features: list[MultipathFeature]
    eps: float
    min_samples: int
    cluster_count: int
    noise_fraction: float


def _feature_matrix(features: list[MultipathFeature]) -> np.ndarray:
    rows = []
    for item in features:
        freq = item.frequency_khz if np.isfinite(item.frequency_khz) else 0.0
        amp = np.clip(item.amplitude_db, -60.0, 0.0)
        rows.append([item.delay_us, freq, amp])
    raw = np.asarray(rows, dtype=float)
    if raw.size == 0:
        return raw.reshape(0, 3)

    delay = np.clip(raw[:, 0] / 700.0, 0.0, 1.0)
    freq = np.clip((raw[:, 1] - 20.0) / 200.0, 0.0, 1.0)
    amp = np.clip((raw[:, 2] + 60.0) / 60.0, 0.0, 1.0)

    weights = np.sqrt(np.asarray([0.5, 0.3, 0.2], dtype=float))
    return np.column_stack([delay, freq, amp]) * weights


def _score(labels: np.ndarray) -> tuple[float, int, float]:
    labeled = labels >= 0
    cluster_ids = sorted(set(labels[labeled]))
    cluster_count = len(cluster_ids)
    noise_fraction = 1.0 - (float(np.count_nonzero(labeled)) / len(labels)) if len(labels) else 1.0
    if not 1 <= cluster_count <= 3:
        return 1_000.0 + abs(cluster_count - 3) * 20.0 + noise_fraction, cluster_count, noise_fraction
    prefer_two_or_three = 0.0 if cluster_count in (2, 3) else 5.0
    return prefer_two_or_three + noise_fraction * 6.0 + abs(cluster_count - 3) * 0.8, cluster_count, noise_fraction


def cluster_multipath_features(features: list[MultipathFeature]) -> ClusteringResult:
    if not features:
        return ClusteringResult(features=[], eps=np.nan, min_samples=0, cluster_count=0, noise_fraction=1.0)

    direct_items = [item for item in features if item.delay_us <= 5.0]
    replica_items = [item for item in features if item.delay_us > 5.0]
    for item in direct_items:
        item.cluster_id = 0
        item.is_direct_path = True

    if not replica_items:
        return ClusteringResult(
            features=features,
            eps=np.nan,
            min_samples=0,
            cluster_count=1 if direct_items else 0,
            noise_fraction=0.0,
        )

    matrix = _feature_matrix(replica_items)
    best: tuple[float, np.ndarray, float, int, int, float] | None = None
    for eps in np.linspace(0.1, 1.0, 19):
        for min_samples in (2, 3):
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(matrix)
            score, cluster_count, noise_fraction = _replica_score(labels)
            if best is None or score < best[0]:
                best = (score, labels, float(eps), min_samples, cluster_count, noise_fraction)

    assert best is not None
    _, labels, eps, min_samples, replica_cluster_count, noise_fraction = best

    med_delay_by_label = {
        label: float(np.median([f.delay_us for f, lab in zip(replica_items, labels) if lab == label]))
        for label in sorted(set(labels))
        if label >= 0
    }
    ordered = {label: i + 1 for i, label in enumerate(sorted(med_delay_by_label, key=med_delay_by_label.get)[:20])}

    for item, label in zip(replica_items, labels):
        item.cluster_id = ordered.get(int(label), -1)
        item.is_direct_path = False

    stable_clusters = {item.cluster_id for item in features if item.cluster_id >= 0}
    return ClusteringResult(
        features=features,
        eps=eps,
        min_samples=min_samples,
        cluster_count=len(stable_clusters),
        noise_fraction=noise_fraction,
    )


def _replica_score(labels: np.ndarray) -> tuple[float, int, float]:
    labeled = labels >= 0
    cluster_count = len(set(labels[labeled]))
    noise_fraction = 1.0 - (float(np.count_nonzero(labeled)) / len(labels)) if len(labels) else 1.0
    if not 1 <= cluster_count <= 20:
        return 1_000.0 + abs(cluster_count - 10) * 20.0 + noise_fraction, cluster_count, noise_fraction
    prefer_many = 0.0 if 6 <= cluster_count <= 20 else 2.0
    return prefer_many + noise_fraction * 2.0 + abs(cluster_count - 10) * 0.2, cluster_count, noise_fraction
