#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune the multipath detector against the manually annotated diagnostic set."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


MATCH_TOLERANCE_US = 50.0
N_FOLDS = 5
RANDOM_SEED = 20260809


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """Load and strictly validate the manual annotation table."""
    annotations = pd.read_csv(annotation_path, encoding="utf-8-sig")
    required = {
        "source_path",
        "relative_path",
        "voyage",
        "pulse_train",
        "click",
        "sample_rate",
        "status",
        "main_sample",
        "multipath1_sample",
        "multipath2_sample",
    }
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"人工标注缺少字段：{missing}")
    if annotations.empty:
        raise ValueError("人工标注表为空")
    if annotations["relative_path"].duplicated().any():
        duplicated = annotations.loc[
            annotations["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise ValueError(f"人工标注存在重复样本：{duplicated[:5]}")
    invalid_status = sorted(set(annotations["status"]) - {"completed", "no_multipath"})
    if invalid_status:
        raise ValueError(f"人工标注包含未完成或未知状态：{invalid_status}")
    missing_files = [source for source in annotations["source_path"] if not Path(source).is_file()]
    if missing_files:
        raise FileNotFoundError(f"人工标注对应的WAV不存在：{missing_files[:5]}")
    if annotations["main_sample"].map(_finite_float).isna().any():
        raise ValueError("人工标注包含缺失的主到达位置")
    completed_missing = annotations["status"].eq("completed") & annotations[
        "multipath1_sample"
    ].map(_finite_float).isna()
    if completed_missing.any():
        raise ValueError("status=completed 的样本存在缺失的第一条多径")
    negative_with_paths = annotations["status"].eq("no_multipath") & (
        annotations["multipath1_sample"].map(_finite_float).notna()
        | annotations["multipath2_sample"].map(_finite_float).notna()
    )
    if negative_with_paths.any():
        raise ValueError("status=no_multipath 的样本仍包含多径位置")
    annotations = annotations.copy()
    annotations["group_id"] = (
        annotations["voyage"].astype(str) + "|" + annotations["pulse_train"].astype(str)
    )
    annotations["true_multipath_count"] = annotations.apply(
        lambda row: sum(
            _finite_float(row[column]) is not None
            for column in ("multipath1_sample", "multipath2_sample")
        ),
        axis=1,
    )
    return annotations.reset_index(drop=True)


def assign_group_folds(annotations: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Create deterministic, approximately stratified group folds."""
    grouped = (
        annotations.groupby("group_id", sort=True)
        .agg(
            sample_count=("relative_path", "size"),
            true_path_count=("true_multipath_count", "sum"),
            no_multipath_count=("status", lambda values: int((values == "no_multipath").sum())),
        )
        .reset_index()
    )
    rng = np.random.default_rng(RANDOM_SEED)
    grouped["tie_break"] = rng.random(len(grouped))
    grouped = grouped.sort_values(
        ["sample_count", "true_path_count", "no_multipath_count", "tie_break"],
        ascending=[False, False, False, True],
    )
    targets = np.array(
        [
            grouped["sample_count"].sum(),
            grouped["true_path_count"].sum(),
            grouped["no_multipath_count"].sum(),
        ],
        dtype=float,
    ) / n_folds
    targets = np.maximum(targets, 1.0)
    fold_totals = np.zeros((n_folds, 3), dtype=float)
    assignment: dict[str, int] = {}
    for row in grouped.itertuples(index=False):
        contribution = np.array(
            [row.sample_count, row.true_path_count, row.no_multipath_count], dtype=float
        )
        scores = []
        for fold in range(n_folds):
            proposed = fold_totals.copy()
            proposed[fold] += contribution
            normalized = (proposed - targets) / targets
            scores.append(float(np.square(normalized).sum()))
        chosen = int(np.argmin(scores))
        assignment[row.group_id] = chosen
        fold_totals[chosen] += contribution
    result = annotations.copy()
    result["fold"] = result["group_id"].map(assignment).astype(int)
    if result.groupby("group_id")["fold"].nunique().max() != 1:
        raise AssertionError("同一脉冲串被拆分到了不同折")
    return result


def _true_path_samples(row: pd.Series) -> list[int]:
    values: list[int] = []
    for column in ("multipath1_sample", "multipath2_sample"):
        value = _finite_float(row[column])
        if value is not None:
            values.append(int(round(value)))
    return sorted(values)


def match_paths(
    predicted: list[int], true: list[int], tolerance_samples: int
) -> tuple[list[tuple[int, int, int]], int, int, int]:
    """Maximise one-to-one matches, then minimise their absolute sample error."""
    predicted = [int(value) for value in predicted]
    true = [int(value) for value in true]
    eligible = [
        (pred_index, true_index, abs(pred_value - true_value))
        for pred_index, pred_value in enumerate(predicted)
        for true_index, true_value in enumerate(true)
        if abs(pred_value - true_value) <= tolerance_samples
    ]
    best_indices: list[tuple[int, int, int]] = []
    if len(predicted) == 2 and len(true) == 2:
        assignments = []
        lookup = {(pred_index, true_index): error for pred_index, true_index, error in eligible}
        for pairs in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
            if all(pair in lookup for pair in pairs):
                assignments.append(
                    [(pred_index, true_index, lookup[(pred_index, true_index)]) for pred_index, true_index in pairs]
                )
        if assignments:
            best_indices = min(assignments, key=lambda pairs: sum(pair[2] for pair in pairs))
    if not best_indices and eligible:
        best_indices = [min(eligible, key=lambda item: item[2])]
    best = [
        (predicted[pred_index], true[true_index], error)
        for pred_index, true_index, error in best_indices
    ]
    tp = len(best)
    return best, tp, len(predicted) - tp, len(true) - tp


def _empty_stats() -> dict[str, object]:
    return {
        "files": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "exact_count": 0,
        "negative_files": 0,
        "true_negative_files": 0,
        "main_within_tolerance": 0,
        "main_abs_errors": [],
        "delay_abs_errors": [],
    }


def _update_stats(
    stats: dict[str, object],
    row: pd.Series,
    predicted_main: int,
    predicted_paths: list[int],
) -> dict[str, object]:
    sample_rate = int(row["sample_rate"])
    tolerance_samples = max(1, int(round(MATCH_TOLERANCE_US * 1e-6 * sample_rate)))
    true_paths = _true_path_samples(row)
    matches, tp, fp, fn = match_paths(predicted_paths, true_paths, tolerance_samples)
    main_error = abs(predicted_main - int(round(float(row["main_sample"]))))
    stats["files"] += 1
    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    stats["exact_count"] += int(len(predicted_paths) == len(true_paths))
    stats["negative_files"] += int(not true_paths)
    stats["true_negative_files"] += int(not true_paths and not predicted_paths)
    stats["main_within_tolerance"] += int(main_error <= tolerance_samples)
    stats["main_abs_errors"].append(main_error)
    stats["delay_abs_errors"].extend(match[2] for match in matches)
    return {
        "true_paths": true_paths,
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "main_error": main_error,
        "tolerance_samples": tolerance_samples,
    }


def _merge_stats(items: list[dict[str, object]]) -> dict[str, object]:
    merged = _empty_stats()
    for item in items:
        for key in (
            "files",
            "tp",
            "fp",
            "fn",
            "exact_count",
            "negative_files",
            "true_negative_files",
            "main_within_tolerance",
        ):
            merged[key] += int(item[key])
        merged["main_abs_errors"].extend(item["main_abs_errors"])
        merged["delay_abs_errors"].extend(item["delay_abs_errors"])
    return merged


def _metrics(stats: dict[str, object], sample_rate: int = 576_000) -> dict[str, float | int]:
    tp, fp, fn = int(stats["tp"]), int(stats["fp"]), int(stats["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    f2 = 5.0 * precision * recall / max(4.0 * precision + recall, np.finfo(float).eps)
    main_errors = np.asarray(stats["main_abs_errors"], dtype=float)
    delay_errors = np.asarray(stats["delay_abs_errors"], dtype=float)

    def samples_to_us(value: float) -> float:
        return 1e6 * value / sample_rate

    return {
        "files": int(stats["files"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_count": int(stats["exact_count"]),
        "negative_files": int(stats["negative_files"]),
        "true_negative_files": int(stats["true_negative_files"]),
        "main_within_tolerance": int(stats["main_within_tolerance"]),
        "delay_abs_error_sum_samples": float(delay_errors.sum()),
        "delay_abs_error_count": int(len(delay_errors)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "exact_count_accuracy": int(stats["exact_count"]) / max(int(stats["files"]), 1),
        "no_multipath_specificity": int(stats["true_negative_files"])
        / max(int(stats["negative_files"]), 1),
        "main_within_50us_rate": int(stats["main_within_tolerance"])
        / max(int(stats["files"]), 1),
        "main_mae_us": samples_to_us(float(np.mean(main_errors))) if len(main_errors) else np.nan,
        "main_median_ae_us": samples_to_us(float(np.median(main_errors)))
        if len(main_errors)
        else np.nan,
        "delay_mae_us": samples_to_us(float(np.mean(delay_errors))) if len(delay_errors) else np.nan,
        "delay_median_ae_us": samples_to_us(float(np.median(delay_errors)))
        if len(delay_errors)
        else np.nan,
        "delay_p95_ae_us": samples_to_us(float(np.quantile(delay_errors, 0.95)))
        if len(delay_errors)
        else np.nan,
    }


def _generator_specs() -> list[dict[str, float]]:
    specs: list[dict[str, float]] = []
    peak_profiles = ((2.0, 1.0), (3.0, 1.5), (4.0, 2.0))
    for segment_us in (25.0, 35.0, 50.0, 75.0):
        for center_width_ms in (0.10, 0.25, 0.50):
            for guard_us in (20.0, 30.0, 40.0, 60.0, 100.0):
                for max_delay_ms in (1.0, 1.5, 2.0):
                    for height, prominence in peak_profiles:
                        specs.append(
                            {
                                "segment_us": segment_us,
                                "center_search_half_width_ms": center_width_ms,
                                "direct_guard_us": guard_us,
                                "max_delay_ms": max_delay_ms,
                                "candidate_min_distance_us": segment_us,
                                "peak_height_mad_factor": height,
                                "peak_prominence_mad_factor": prominence,
                            }
                        )
    return specs


GATE_PROFILES = (
    ("very_loose", 0.0, 0.20, 0.25, 0.25),
    ("loose", 0.0, 0.30, 0.35, 0.30),
    ("recall_1", 2.0, 0.30, 0.40, 0.35),
    ("recall_2", 2.0, 0.40, 0.45, 0.40),
    ("balanced", 4.0, 0.45, 0.50, 0.45),
    ("current_recall", 6.0, 0.50, 0.60, 0.55),
    ("strict", 6.0, 0.55, 0.70, 0.60),
)


WEIGHT_PROFILES = (
    ("energy20_ncc50_spectral30", 0.20, 0.50, 0.30),
    ("energy10_ncc60_spectral30", 0.10, 0.60, 0.30),
    ("energy20_ncc60_spectral20", 0.20, 0.60, 0.20),
    ("energy30_ncc40_spectral30", 0.30, 0.40, 0.30),
)


def _select_from_candidates(
    candidates: pd.DataFrame,
    sample_rate: int,
    config: object,
) -> list[int]:
    if candidates.empty:
        return []
    table = candidates.copy()
    table["score"] = (
        config.weight_energy * table["energy_score"]
        + config.weight_ncc * table["abs_ncc"]
        + config.weight_spectral * table["spectral_cosine"]
    )
    table["passes_thresholds"] = (
        table["candidate_source"].eq("mad_peak")
        & table["snr_db"].ge(config.min_candidate_snr_db)
        & table["abs_ncc"].ge(config.min_abs_ncc)
        & table["spectral_cosine"].ge(config.min_spectral_cosine)
        & table["score"].ge(config.min_accept_score)
    )
    eligible = table[table["passes_thresholds"]].sort_values("score", ascending=False)
    segment_samples = max(1, int(round(config.segment_us * 1e-6 * sample_rate)))
    centers: list[int] = []
    for center in eligible["center_sample"].astype(int):
        if any(abs(center - previous) < segment_samples for previous in centers):
            continue
        centers.append(int(center))
        if len(centers) >= int(config.max_multipaths):
            break
    return sorted(centers)


def _candidate_arrays(candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a small candidate table once for repeated threshold evaluation."""
    if candidates.empty:
        return {
            "center": np.empty(0, dtype=np.int64),
            "mad_peak": np.empty(0, dtype=bool),
            "snr": np.empty(0, dtype=float),
            "ncc": np.empty(0, dtype=float),
            "spectral": np.empty(0, dtype=float),
            "energy": np.empty(0, dtype=float),
        }
    return {
        "center": candidates["center_sample"].to_numpy(dtype=np.int64, copy=True),
        "mad_peak": candidates["candidate_source"].eq("mad_peak").to_numpy(dtype=bool),
        "snr": candidates["snr_db"].to_numpy(dtype=float, copy=True),
        "ncc": candidates["abs_ncc"].to_numpy(dtype=float, copy=True),
        "spectral": candidates["spectral_cosine"].to_numpy(dtype=float, copy=True),
        "energy": candidates["energy_score"].to_numpy(dtype=float, copy=True),
    }


def _select_from_arrays(
    arrays: dict[str, np.ndarray], sample_rate: int, config: object
) -> list[int]:
    """Fast equivalent of detector thresholding and greedy non-overlap selection."""
    if not len(arrays["center"]):
        return []
    scores = (
        config.weight_energy * arrays["energy"]
        + config.weight_ncc * arrays["ncc"]
        + config.weight_spectral * arrays["spectral"]
    )
    eligible = np.flatnonzero(
        arrays["mad_peak"]
        & (arrays["snr"] >= config.min_candidate_snr_db)
        & (arrays["ncc"] >= config.min_abs_ncc)
        & (arrays["spectral"] >= config.min_spectral_cosine)
        & (scores >= config.min_accept_score)
    )
    if not len(eligible):
        return []
    order = eligible[np.argsort(scores[eligible])[::-1]]
    segment_samples = max(1, int(round(config.segment_us * 1e-6 * sample_rate)))
    centers: list[int] = []
    for index in order:
        center = int(arrays["center"][index])
        if any(abs(center - previous) < segment_samples for previous in centers):
            continue
        centers.append(center)
        if len(centers) >= int(config.max_multipaths):
            break
    return sorted(centers)


def _update_stats_values(
    stats: dict[str, object],
    sample_rate: int,
    true_main: int,
    true_paths: list[int],
    predicted_main: int,
    predicted_paths: list[int],
) -> None:
    """Update aggregate metrics without constructing a Pandas row."""
    tolerance_samples = max(1, int(round(MATCH_TOLERANCE_US * 1e-6 * sample_rate)))
    matches, tp, fp, fn = match_paths(predicted_paths, true_paths, tolerance_samples)
    main_error = abs(predicted_main - true_main)
    stats["files"] += 1
    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    stats["exact_count"] += int(len(predicted_paths) == len(true_paths))
    stats["negative_files"] += int(not true_paths)
    stats["true_negative_files"] += int(not true_paths and not predicted_paths)
    stats["main_within_tolerance"] += int(main_error <= tolerance_samples)
    stats["main_abs_errors"].append(main_error)
    stats["delay_abs_errors"].extend(match[2] for match in matches)


def _selection_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    delay_mae = float(metrics["delay_mae_us"])
    if not np.isfinite(delay_mae):
        delay_mae = np.inf
    return (
        float(metrics["f2"]),
        float(metrics["precision"]),
        -delay_mae,
        float(metrics["exact_count_accuracy"]),
        float(metrics["no_multipath_specificity"]),
    )


def _config_from_record(base_config: object, record: pd.Series | dict[str, object]) -> object:
    fields = set(asdict(base_config))
    values = {key: record[key] for key in fields if key in record}
    values["max_multipaths"] = int(values.get("max_multipaths", 2))
    return replace(base_config, **values)


def _predict_exact(
    row: pd.Series,
    signals: dict[str, tuple[int, np.ndarray]],
    config: object,
    detect_paths: Callable,
) -> tuple[int, list[int]]:
    sample_rate, signal = signals[row["source_path"]]
    paths, _, _ = detect_paths(signal, sample_rate, config)
    main = int(paths.iloc[0]["center_sample"])
    multipaths = (
        paths.loc[paths["label"].ne("main_arrival"), "center_sample"].astype(int).tolist()
    )
    return main, sorted(multipaths)


def _evaluate_exact_config(
    annotations: pd.DataFrame,
    signals: dict[str, tuple[int, np.ndarray]],
    config: object,
    detect_paths: Callable,
    progress: tqdm,
    selected_by_fold: dict[int, str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    stats = _empty_stats()
    rows: list[dict[str, object]] = []
    for _, row in annotations.iterrows():
        predicted_main, predicted_paths = _predict_exact(row, signals, config, detect_paths)
        details = _update_stats(stats, row, predicted_main, predicted_paths)
        matches = details["matches"]
        rows.append(
            {
                "source_path": row["source_path"],
                "relative_path": row["relative_path"],
                "voyage": row["voyage"],
                "pulse_train": row["pulse_train"],
                "click": row["click"],
                "fold": int(row["fold"]),
                "selected_config": selected_by_fold.get(int(row["fold"]), "final_full_fit")
                if selected_by_fold
                else "final_full_fit",
                "true_main_sample": int(round(float(row["main_sample"]))),
                "predicted_main_sample": predicted_main,
                "main_abs_error_samples": details["main_error"],
                "true_multipath_samples": json.dumps(details["true_paths"]),
                "predicted_multipath_samples": json.dumps(predicted_paths),
                "matched_sample_pairs": json.dumps(
                    [[int(pred), int(true), int(error)] for pred, true, error in matches]
                ),
                "tp": details["tp"],
                "fp": details["fp"],
                "fn": details["fn"],
                "exact_count": int(len(predicted_paths) == len(details["true_paths"])),
            }
        )
        progress.update(1)
    return stats, pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                text = f"{float(value):.6g}"
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    annotations: pd.DataFrame,
    fold_summary: pd.DataFrame,
    oof_metrics: dict[str, object],
    final_metrics: dict[str, object],
    final_config: object,
    config_count: int,
) -> None:
    fold_distribution = (
        annotations.groupby("fold")
        .agg(
            files=("relative_path", "size"),
            groups=("group_id", "nunique"),
            true_paths=("true_multipath_count", "sum"),
            no_multipath=("status", lambda values: int((values == "no_multipath").sum())),
        )
        .reset_index()
    )
    lines = [
        "# 人工标注集多径参数调优与验证报告",
        "",
        "## 数据与验证设计",
        "",
        f"- 有效人工标注：{len(annotations)}条。",
        f"- 有多径：{int(annotations['true_multipath_count'].gt(0).sum())}条；无多径：{int(annotations['true_multipath_count'].eq(0).sum())}条。",
        f"- 人工标注多径总数：{int(annotations['true_multipath_count'].sum())}条。",
        f"- 参数候选：{config_count}组。",
        f"- 验证：按航次与脉冲串分组的{N_FOLDS}折交叉验证，同一脉冲串不跨折。",
        f"- 匹配判据：检测位置与人工位置相差不超过±{MATCH_TOLERANCE_US:g} μs；一对一匹配。",
        "- 每个外层验证折只使用其余四折选择F2最高的参数；同分依次比较Precision、时延MAE、路径数量完全正确率和无多径识别率。",
        "",
        "## 折外验证结果",
        "",
        "```json",
        json.dumps(oof_metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "折外结果由每一折各自在其训练部分选出的参数产生，是本次主要效果估计。",
        "",
        "## 结果解释与使用边界",
        "",
        f"- 主到达定位较稳定：±50 μs内比例为{oof_metrics['main_within_50us_rate']:.2%}，MAE为{oof_metrics['main_mae_us']:.3f} μs。",
        f"- 多径定位在已匹配路径上较准确：时延MAE为{oof_metrics['delay_mae_us']:.3f} μs，P95绝对误差为{oof_metrics['delay_p95_ae_us']:.3f} μs。",
        f"- 路径级召回率为{oof_metrics['recall']:.2%}，但Precision为{oof_metrics['precision']:.2%}，路径数量完全正确率为{oof_metrics['exact_count_accuracy']:.2%}。",
        f"- 无多径识别率仅为{oof_metrics['no_multipath_specificity']:.2%}（{int(oof_metrics['true_negative_files'])}/{int(oof_metrics['negative_files'])}）；F2优先参数存在明显过检。",
        "- 因此该参数适合作为高召回候选生成器；若全量结果要直接作为最终多径标签，应增加人工复核或改用更强调Precision/无多径识别率的选参目标。",
        "",
        "## 各折选参",
        "",
        _markdown_table(fold_summary),
        "",
        "## 折分布",
        "",
        _markdown_table(fold_distribution),
        "",
        "## 后续全量应用参数",
        "",
        "以下参数使用全部199条标注选择，供后续应用到3500条全量样本；其全标注集指标属于回代结果，不作为独立验证分数。",
        "",
        "```json",
        json.dumps(asdict(final_config), ensure_ascii=False, indent=2),
        "```",
        "",
        "全标注集回代指标：",
        "",
        "```json",
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        "```",
    ]
    (output_dir / "manual_annotation_tuning_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_manual_annotation_tuning(
    annotation_path: Path,
    output_dir: Path,
    base_config: object,
    prepare_signals: Callable,
    detect_paths: Callable,
) -> dict[str, object]:
    """Run grouped CV tuning and write reproducible diagnostics-only outputs."""
    annotation_path = Path(annotation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = assign_group_folds(load_annotations(annotation_path))
    generator_specs = _generator_specs()
    config_count = len(generator_specs) * len(GATE_PROFILES) * len(WEIGHT_PROFILES)
    total_steps = (
        len(annotations)
        + len(generator_specs) * len(annotations)
        + config_count
        + 2 * len(annotations)
    )
    progress = tqdm(total=total_steps, desc="总体进度", unit="step", dynamic_ncols=True)
    signals: dict[str, tuple[int, np.ndarray]] = {}
    try:
        for row in annotations.itertuples(index=False):
            sample_rate, _, signal = prepare_signals(Path(row.source_path))
            signals[row.source_path] = (int(sample_rate), signal)
            progress.update(1)

        truth_records = [
            (
                int(row.fold),
                int(row.sample_rate),
                int(round(float(row.main_sample))),
                sorted(
                    int(round(value))
                    for value in (row.multipath1_sample, row.multipath2_sample)
                    if _finite_float(value) is not None
                ),
            )
            for row in annotations.itertuples(index=False)
        ]

        metric_records: list[dict[str, object]] = []
        config_index = 0
        for generator_index, spec in enumerate(generator_specs):
            generator_config = replace(
                base_config,
                **spec,
                min_candidate_snr_db=0.0,
                min_abs_ncc=0.0,
                min_spectral_cosine=0.0,
                min_accept_score=0.0,
                weight_energy=0.20,
                weight_ncc=0.50,
                weight_spectral=0.30,
                max_multipaths=2,
            )
            generated: list[tuple[int, dict[str, np.ndarray]]] = []
            main_centers: list[int] = []
            for row in annotations.itertuples(index=False):
                sample_rate, signal = signals[row.source_path]
                paths, candidates, _ = detect_paths(signal, sample_rate, generator_config)
                main_centers.append(int(paths.iloc[0]["center_sample"]))
                generated.append((sample_rate, _candidate_arrays(candidates)))
                progress.update(1)

            for gate_name, snr, ncc, spectral, accept in GATE_PROFILES:
                for weight_name, weight_energy, weight_ncc, weight_spectral in WEIGHT_PROFILES:
                    config = replace(
                        generator_config,
                        min_candidate_snr_db=snr,
                        min_abs_ncc=ncc,
                        min_spectral_cosine=spectral,
                        min_accept_score=accept,
                        weight_energy=weight_energy,
                        weight_ncc=weight_ncc,
                        weight_spectral=weight_spectral,
                    )
                    fold_stats = [_empty_stats() for _ in range(N_FOLDS)]
                    for row_index, truth in enumerate(truth_records):
                        fold, sample_rate, true_main, true_paths = truth
                        _, candidate_values = generated[row_index]
                        predicted = _select_from_arrays(candidate_values, sample_rate, config)
                        _update_stats_values(
                            fold_stats[fold],
                            sample_rate,
                            true_main,
                            true_paths,
                            main_centers[row_index],
                            predicted,
                        )
                    total_stats = _merge_stats(fold_stats)
                    total_metrics = _metrics(total_stats)
                    config_name = f"config_{config_index:05d}"
                    record: dict[str, object] = {
                        "config_name": config_name,
                        "generator_index": generator_index,
                        "gate_profile": gate_name,
                        "weight_profile": weight_name,
                        **asdict(config),
                    }
                    record.update({f"all_{key}": value for key, value in total_metrics.items()})
                    for fold, stats in enumerate(fold_stats):
                        fold_metrics = _metrics(stats)
                        for key, value in fold_metrics.items():
                            record[f"fold_{fold}_{key}"] = value
                    metric_records.append(record)
                    config_index += 1
                    progress.update(1)

        metrics_table = pd.DataFrame(metric_records)
        selected_records: list[dict[str, object]] = []
        selected_configs: dict[int, object] = {}
        selected_names: dict[int, str] = {}
        for validation_fold in range(N_FOLDS):
            best_index = None
            best_key = None
            best_train_metrics = None
            for index, record in metrics_table.iterrows():
                tp = int(record["all_tp"] - record[f"fold_{validation_fold}_tp"])
                fp = int(record["all_fp"] - record[f"fold_{validation_fold}_fp"])
                fn = int(record["all_fn"] - record[f"fold_{validation_fold}_fn"])
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
                f2 = 5.0 * precision * recall / max(
                    4.0 * precision + recall, np.finfo(float).eps
                )
                files = int(record["all_files"] - record[f"fold_{validation_fold}_files"])
                exact_count = int(
                    record["all_exact_count"]
                    - record[f"fold_{validation_fold}_exact_count"]
                )
                negative_files = int(
                    record["all_negative_files"]
                    - record[f"fold_{validation_fold}_negative_files"]
                )
                true_negative_files = int(
                    record["all_true_negative_files"]
                    - record[f"fold_{validation_fold}_true_negative_files"]
                )
                delay_error_sum = float(
                    record["all_delay_abs_error_sum_samples"]
                    - record[f"fold_{validation_fold}_delay_abs_error_sum_samples"]
                )
                delay_error_count = int(
                    record["all_delay_abs_error_count"]
                    - record[f"fold_{validation_fold}_delay_abs_error_count"]
                )
                train_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f2": f2,
                    "delay_mae_us": 1e6
                    * delay_error_sum
                    / max(delay_error_count, 1)
                    / 576_000,
                    "exact_count_accuracy": exact_count / max(files, 1),
                    "no_multipath_specificity": true_negative_files
                    / max(negative_files, 1),
                }
                key = _selection_key(train_metrics)
                if best_key is None or key > best_key:
                    best_index = index
                    best_key = key
                    best_train_metrics = train_metrics
            if best_index is None:
                raise RuntimeError(f"第{validation_fold}折没有可用参数")
            best_record = metrics_table.loc[best_index]
            config = _config_from_record(base_config, best_record)
            selected_configs[validation_fold] = config
            selected_names[validation_fold] = str(best_record["config_name"])
            selected_records.append(
                {
                    "validation_fold": validation_fold,
                    "config_name": best_record["config_name"],
                    "training_f2": best_train_metrics["f2"],
                    "training_precision": best_train_metrics["precision"],
                    "training_recall": best_train_metrics["recall"],
                    "validation_f2": best_record[f"fold_{validation_fold}_f2"],
                    "validation_precision": best_record[f"fold_{validation_fold}_precision"],
                    "validation_recall": best_record[f"fold_{validation_fold}_recall"],
                    **asdict(config),
                }
            )

        oof_stats = _empty_stats()
        oof_rows: list[dict[str, object]] = []
        for _, row in annotations.iterrows():
            fold = int(row["fold"])
            predicted_main, predicted_paths = _predict_exact(
                row, signals, selected_configs[fold], detect_paths
            )
            details = _update_stats(oof_stats, row, predicted_main, predicted_paths)
            oof_rows.append(
                {
                    "source_path": row["source_path"],
                    "relative_path": row["relative_path"],
                    "voyage": row["voyage"],
                    "pulse_train": row["pulse_train"],
                    "click": row["click"],
                    "fold": fold,
                    "selected_config": selected_names[fold],
                    "true_main_sample": int(round(float(row["main_sample"]))),
                    "predicted_main_sample": predicted_main,
                    "main_abs_error_samples": details["main_error"],
                    "true_multipath_samples": json.dumps(details["true_paths"]),
                    "predicted_multipath_samples": json.dumps(predicted_paths),
                    "matched_sample_pairs": json.dumps(
                        [[int(pred), int(true), int(error)] for pred, true, error in details["matches"]]
                    ),
                    "tp": details["tp"],
                    "fp": details["fp"],
                    "fn": details["fn"],
                    "exact_count": int(len(predicted_paths) == len(details["true_paths"])),
                }
            )
            progress.update(1)
        oof_metrics = _metrics(oof_stats)

        final_index = max(
            metrics_table.index,
            key=lambda index: _selection_key(
                {
                    key.removeprefix("all_"): value
                    for key, value in metrics_table.loc[index].items()
                    if key.startswith("all_")
                }
            ),
        )
        final_record = metrics_table.loc[final_index]
        final_config = _config_from_record(base_config, final_record)
        final_stats, final_predictions = _evaluate_exact_config(
            annotations,
            signals,
            final_config,
            detect_paths,
            progress,
        )
        final_metrics = _metrics(final_stats)

        annotations.to_csv(output_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")
        metrics_table.to_csv(
            output_dir / "parameter_cv_metrics.csv", index=False, encoding="utf-8-sig"
        )
        fold_summary = pd.DataFrame(selected_records)
        fold_summary.to_csv(
            output_dir / "fold_selected_parameters.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(oof_rows).to_csv(
            output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
        )
        final_predictions.to_csv(
            output_dir / "final_fit_predictions.csv", index=False, encoding="utf-8-sig"
        )
        (output_dir / "oof_metrics.json").write_text(
            json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "final_fit_metrics.json").write_text(
            json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "final_parameters.json").write_text(
            json.dumps(asdict(final_config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "annotation_path": str(annotation_path.resolve()),
            "annotation_rows": len(annotations),
            "folds": N_FOLDS,
            "group_columns": ["voyage", "pulse_train"],
            "match_tolerance_us": MATCH_TOLERANCE_US,
            "selection_metric": "path_level_f2",
            "tie_breakers": [
                "precision",
                "delay_mae_us",
                "exact_count_accuracy",
                "no_multipath_specificity",
            ],
            "candidate_config_count": config_count,
            "final_config_name": final_record["config_name"],
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_report(
            output_dir,
            annotations,
            fold_summary,
            oof_metrics,
            final_metrics,
            final_config,
            config_count,
        )
    finally:
        progress.close()
    return {
        "annotations": annotations,
        "fold_summary": fold_summary,
        "oof_predictions": pd.DataFrame(oof_rows),
        "oof_metrics": oof_metrics,
        "parameter_metrics": metrics_table,
        "final_config": final_config,
        "final_fit_metrics": final_metrics,
        "final_fit_predictions": final_predictions,
        "output_dir": output_dir,
    }
