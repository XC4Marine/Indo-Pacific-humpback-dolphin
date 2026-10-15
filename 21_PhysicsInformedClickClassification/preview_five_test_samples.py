"""Infer five held-out samples and export waveforms, labels, and a row-wise figure."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pipeline import PhysicalClassifier


def main():
    experiment = Path(__file__).resolve().parent
    root = experiment / "outputs_snr_m15_p15_full_5seeds"
    output = experiment / "five_test_samples"
    output.mkdir(exist_ok=True)
    seed, fold = 20260826, 0
    condition = "recon_physics_bce"
    with tqdm(total=5, desc="Overall progress", unit="stage") as progress:
        config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        manifest = pd.read_csv(root / "dataset_manifest.csv")
        predictions = pd.read_csv(root / "predictions" / "sample_predictions.csv")
        candidates = predictions.loc[
            predictions.condition.eq(condition) & predictions.seed.eq(seed)
            & predictions.test_fold.eq(fold)
        ].sort_values("sample_index")
        # Fixed random selection within each class, independent of prediction quality.
        selected = pd.concat([
            candidates.loc[candidates.label.eq(1)].sample(n=3, random_state=seed),
            candidates.loc[candidates.label.eq(0)].sample(n=2, random_state=seed),
        ]).reset_index(drop=True)
        indices = selected.sample_index.to_numpy(dtype=int)
        rows = manifest.iloc[indices]
        assert np.array_equal(rows.sample_id.to_numpy(), selected.sample_id.to_numpy())
        assert rows.test_fold.eq(fold).all()
        assert np.array_equal(rows.label.to_numpy(), selected.label.to_numpy())
        waveforms = np.load(root / "cache" / "preprocessed_waveforms.npy", mmap_mode="r")
        y = np.array(waveforms[indices], dtype=np.float32, copy=True)
        progress.update()

        checkpoint = root / "models" / f"{condition}_seed_{seed}_fold_{fold}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["fold"] == fold and saved["seed"] == seed
        model = PhysicalClassifier(
            sample_count=config["sample_count"], response_count=config["response_count"],
            include_residual=saved.get("include_residual_feature", True),
            include_response_feature=saved.get("include_response_feature", True),
        )
        model.load_state_dict(saved["state_dict"])
        model.eval()
        progress.update()
        with torch.inference_mode():
            values = model(torch.from_numpy(y))
        arrays = {name: tensor.numpy() for name, tensor in values.items()}
        probability = torch.sigmoid(values["logit"]).numpy()
        np.testing.assert_allclose(probability, selected.probability.to_numpy(), atol=2e-5, rtol=2e-5)
        for name, values_array in arrays.items():
            assert np.isfinite(values_array).all(), name
        selected["probability_saved"] = selected["probability"]
        selected["probability"] = probability
        selected["predicted_label"] = (probability >= selected.threshold.to_numpy()).astype(int)
        selected["true_label_name"] = np.where(selected.label.eq(1), "dolphin_click", "background_noise")
        selected["predicted_label_name"] = np.where(selected.predicted_label.eq(1), "dolphin_click", "background_noise")
        selected["reconstruction_nrmse"] = np.sqrt(
            np.sum((arrays["reconstruction"] - y) ** 2, axis=1) / np.maximum(np.sum(y ** 2, axis=1), 1e-12)
        )
        progress.update()

        selected.to_csv(output / "sample_labels.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(
            output / "sample_waveforms.npz", x_pred=arrays["source"], h_pred=arrays["response"],
            y_pred=arrays["reconstruction"], y_true=y, sample_id=selected.sample_id.to_numpy(dtype=str),
            label=selected.label.to_numpy(), predicted_label=selected.predicted_label.to_numpy(),
            probability=probability, sample_rate=config["sample_rate"],
        )
        time = np.arange(y.shape[1]) * 1000 / config["sample_rate"]
        delay = np.arange(arrays["response"].shape[1]) * 1000 / config["sample_rate"]
        for i, row in selected.iterrows():
            pd.DataFrame({"time_ms": time, "x_pred": arrays["source"][i],
                          "y_pred": arrays["reconstruction"][i], "y_true": y[i]}).to_csv(
                output / f"sample_{i + 1}_waveforms.csv", index=False)
            pd.DataFrame({"delay_ms": delay, "h_pred": arrays["response"][i]}).to_csv(
                output / f"sample_{i + 1}_response.csv", index=False)
        progress.update()

        plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
        fig, axes = plt.subplots(5, 4, figsize=(19, 14))
        titles = ["预测 x：信源波形", "预测 h：信道响应", "预测 y：x 与 h 的因果卷积", "真实 y 与预测 y 对比"]
        names = {0: "背景噪声", 1: "海豚 Click"}
        for i, row in selected.iterrows():
            axes[i, 0].plot(time, arrays["source"][i], color="#247aab", lw=.7)
            axes[i, 1].plot(delay, arrays["response"][i], color="#8862b2", lw=.8)
            inset = axes[i, 1].inset_axes([.34, .42, .63, .52])
            inset.plot(delay[1:], arrays["response"][i, 1:], color="#8862b2", lw=.65)
            inset.text(.03, .96, "尾部放大（不含首点）", transform=inset.transAxes, va="top", fontsize=7)
            inset.tick_params(labelsize=7)
            axes[i, 2].plot(time, arrays["reconstruction"][i], color="#d36d27", lw=.7)
            axes[i, 3].plot(time, y[i], color="#243746", lw=.8, label="真实 y", alpha=.85)
            axes[i, 3].plot(time, arrays["reconstruction"][i], color="#e78635", lw=.7, label="预测 y", alpha=.8)
            limit = max(np.max(np.abs(y[i])), np.max(np.abs(arrays["reconstruction"][i]))) * 1.08
            axes[i, 2].set_ylim(-limit, limit)
            axes[i, 3].set_ylim(-limit, limit)
            axes[i, 3].legend(loc="upper right", fontsize=8)
            snr = f"\nSNR={row.target_snr_db:.1f} dB" if row.label == 1 else "\n噪声类型：多径噪声"
            display_id = str(row.sample_id).replace("_Type_3_Multipath_Noise", "")
            text = (f"样本 {i + 1}  {display_id}\n"
                    f"真实标签 {int(row.label)}：{names[row.label]}\n"
                    f"预测标签 {int(row.predicted_label)}：{names[row.predicted_label]}\n"
                    f"P(Click)={row.probability:.4f}\n阈值={row.threshold:.4f}{snr}")
            axes[i, 0].text(-1.12, .5, text, transform=axes[i, 0].transAxes, va="center", fontsize=9)
            for j, ax in enumerate(axes[i]):
                ax.set_title(titles[j] if i == 0 else "", fontsize=12)
                ax.set_xlabel("延迟 / ms" if j == 1 else "时间 / ms")
                ax.grid(alpha=.18)
                ax.tick_params(labelsize=8)
        fig.suptitle("实验21 · 测试集5个样本的分解、重建与类别预测", fontsize=18, y=.984)
        fig.text(.5, .949, "Recon + Physics + BCE | seed=20260826 | test fold=0 | 固定随机抽取3个正样本、2个负样本", ha="center", fontsize=11)
        fig.text(.5, .014, "真实 y 为模型输入的预处理观测波形（正样本含固定混合噪声）；x、h 为模型估计。h[0] 固定为1。", ha="center", fontsize=11)
        fig.subplots_adjust(left=.205, right=.985, top=.91, bottom=.06, hspace=.44, wspace=.24)
        fig.savefig(output / "five_test_samples.png", dpi=180)
        fig.savefig(output / "five_test_samples.pdf")
        plt.close(fig)
        metadata = {"checkpoint": str(checkpoint), "seed": seed, "test_fold": fold,
                    "selection": "Fixed random seed; 3 positives and 2 negatives; no filtering on predictions",
                    "y_true": "Preprocessed observed model input, including fixed noise mixing for positives",
                    "probability_matches_saved_predictions": True}
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        progress.update()
    print(selected[["sample_id", "label", "predicted_label", "probability", "threshold", "reconstruction_nrmse"]].to_string(index=False))
    print(output / "five_test_samples.png")


if __name__ == "__main__":
    main()
