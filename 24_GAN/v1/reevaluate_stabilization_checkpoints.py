#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""以固定噪声复评固定第 1 折稳定化检查点，消除检查点间随机采样差异。"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from click_cgan_stabilized import StabilizationConfig, evaluate, make_paths
from click_cgan_v1 import Generator, five_fold_splits, json_dump, load_sampled_dataset, resolve_device, seed_everything


def main() -> None:
    config = StabilizationConfig()
    seed_everything(config.seed)
    paths = make_paths(config)
    device = resolve_device(config.device)
    data_config = type("DataConfig", (), {"data_root": config.data_root, "sample_count": config.sample_count,
                                            "seed": config.seed})()
    waveforms, features, _ = load_sampled_dataset(data_config, paths)
    split_config = type("SplitConfig", (), {"seed": config.seed, "fold_count": config.fold_count})()
    _, _, test_waveforms, test_features, *_ = list(five_fold_splits(waveforms, features, split_config))[0]

    results = []
    for checkpoint_path in sorted(paths["checkpoints"].glob("epoch_*.pt"), key=lambda path: int(re.search(r"\d+", path.stem).group())):
        epoch = int(re.search(r"\d+", checkpoint_path.stem).group())
        generator = Generator().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        metrics = evaluate(generator, test_waveforms, test_features, device, paths, config.seed + 1,
                           f"fixed_epoch_{epoch}")
        metrics["epoch"] = epoch
        results.append(metrics)
        print(f"epoch={epoch}: {metrics}")
    json_dump(paths["metrics"] / "fixed_noise_checkpoint_metrics.json", results)


if __name__ == "__main__":
    main()
