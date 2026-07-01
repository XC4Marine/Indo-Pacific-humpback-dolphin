# 正负样本数据与音频读取说明

本文件夹保存了正负样本的索引表副本。注意：这里复制的是表格文件，实际 WAV 音频仍在原始数据文件夹中。较大的数据文件已写入 `.gitignore`，因此 GitHub 上可能只同步本说明文档；批量读取时以本地路径为准。

## 1. 本文件夹中的表格

正样本表：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\14\pulse_params.txt
```

来源：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\11_ChannelMetrics\pulse_params.txt
```

负样本表：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\14\negative_samples_log.csv
```

来源：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\13_Negative_Bandpass_20k_190k_NoHFER\negative_samples_log.csv
```

说明：请求中写的是 `nagative_samples_log.csv`，项目中实际文件名是 `negative_samples_log.csv`。

## 2. 实际音频路径

### 正样本音频

正样本 WAV 存放在：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate
```

路径结构：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\PulseTrain_XXX\Click_YYY\Pulse_YYY.wav
```

例子：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\PulseTrain_001\Click_001\Pulse_001.wav
```

正样本表 `pulse_params.txt` 中的 `PulseTrain` 和 `ClickName` 可以直接构造音频路径。`Click_001` 对应 `Pulse_001.wav`。

### 负样本音频

负样本 WAV 存放在：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\13_Negative_Bandpass_20k_190k_NoHFER\Negative_Samples
```

路径结构大致为：

```text
原始录音名_时间戳s_噪声类型.wav
```

例子：

```text
D:\Project_Github\Indo-Pacific-humpback-dolphin\13_Negative_Bandpass_20k_190k_NoHFER\Negative_Samples\Ori_Recording_01_0p004597s_Type_4_Isolated_Noise.wav
```

负样本表 `negative_samples_log.csv` 中包含 `Audio_File`、`Noise_Timestamp(s)` 和 `Noise_Type`。多数文件名可由这些列构造，但有少量样本会因为浮点保留位数差 `0.000001` 秒。因此批量读取时建议先扫描 `Negative_Samples` 文件夹建立索引，再按表格逐行匹配。

## 3. 表格结构

### pulse_params.txt

格式：制表符分隔，读取时使用 `sep="\t"`。

字段：

| 字段 | 含义 |
| --- | --- |
| `PulseTrain` | 脉冲串编号，例如 `PulseTrain_001`。 |
| `ClickName` | click 编号，例如 `Click_001`。 |
| `PathType` | 路径类型，`Direct` 为直达声，`Multipath_1` 等为多径。 |
| `Delay_us` | 相对直达声的延迟，单位为微秒。 |
| `AmpRatio` | 直达声 RMS / 当前路径 RMS。 |
| `Coherence` | 当前路径与直达声窗口的 Pearson 相干性。 |
| `ClickClass` | click 级标签，`multipath` 表示该 click 有多径，`isolate` 表示只有直达声。 |

### negative_samples_log.csv

格式：逗号分隔，读取时使用普通 `pd.read_csv()`。

字段：

| 字段 | 含义 |
| --- | --- |
| `Audio_File` | 原始录音文件名。 |
| `Noise_Timestamp(s)` | 负样本在原始录音中的时间戳，单位为秒。 |
| `Centroid` | 负样本频谱质心。 |
| `Noise_Type` | 负样本类型，例如 `Type_4_Isolated_Noise`。 |

## 4. 批量读取表格和音频

下面示例使用 `scipy.io.wavfile.read` 读取 WAV。返回的 `signal` 是 numpy 数组，`sample_rate` 是采样率。

```python
from pathlib import Path
import re

import pandas as pd
from scipy.io import wavfile

repo = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")

positive_table = repo / "14" / "pulse_params.txt"
negative_table = repo / "14" / "negative_samples_log.csv"

positive_audio_root = repo / "09_Locate"
negative_audio_root = repo / "13_Negative_Bandpass_20k_190k_NoHFER" / "Negative_Samples"

positive_df = pd.read_csv(positive_table, sep="\t")
negative_df = pd.read_csv(negative_table)
```

### 读取正样本音频

`pulse_params.txt` 是路径级表格，同一个 click 可能有多行，例如一行 `Direct` 和一行或多行 `Multipath_*`。读取音频时应先去重到 click 级别。

```python
def positive_wav_path(row):
    click_no = row["ClickName"].split("_")[1]
    return positive_audio_root / row["PulseTrain"] / row["ClickName"] / f"Pulse_{click_no}.wav"

positive_clicks = positive_df.drop_duplicates(["PulseTrain", "ClickName"]).copy()
positive_clicks["wav_path"] = positive_clicks.apply(positive_wav_path, axis=1)
positive_clicks["label"] = 1

for row in positive_clicks.itertuples(index=False):
    sample_rate, signal = wavfile.read(row.wav_path)
    # 在这里处理 signal
```

### 读取负样本音频

负样本 WAV 文件名包含时间戳。为了避免浮点小数导致的 1 微秒级命名差异，先扫描文件夹建立索引，再匹配 CSV。

```python
negative_name_re = re.compile(
    r"^(?P<audio>Ori_Recording_\d+)_(?P<time>\d+p\d{6})s_(?P<noise_type>.+)\.wav$"
)

negative_index = {}
for wav_path in negative_audio_root.glob("*.wav"):
    match = negative_name_re.match(wav_path.name)
    if not match:
        continue
    audio_file = match.group("audio") + ".wav"
    timestamp = float(match.group("time").replace("p", "."))
    noise_type = match.group("noise_type")
    key = (audio_file, noise_type)
    negative_index.setdefault(key, []).append((timestamp, wav_path))

for values in negative_index.values():
    values.sort(key=lambda item: item[0])

def negative_wav_path(row, tolerance=2e-6):
    key = (row["Audio_File"], row["Noise_Type"])
    target_time = float(row["Noise_Timestamp(s)"])
    candidates = negative_index.get(key, [])
    if not candidates:
        raise FileNotFoundError(f"No WAV candidates for {key}")
    nearest_time, nearest_path = min(candidates, key=lambda item: abs(item[0] - target_time))
    if abs(nearest_time - target_time) > tolerance:
        raise FileNotFoundError(f"No close WAV for {key} at {target_time}")
    return nearest_path

negative_df["wav_path"] = negative_df.apply(negative_wav_path, axis=1)
negative_df["label"] = 0

for row in negative_df.itertuples(index=False):
    sample_rate, signal = wavfile.read(row.wav_path)
    # 在这里处理 signal
```

## 5. 合并成统一读取队列

如果训练模型需要统一的读取队列，可以只保留 `wav_path` 和 `label`。

```python
positive_queue = positive_clicks[["wav_path", "label"]]
negative_queue = negative_df[["wav_path", "label"]]

all_samples = pd.concat([positive_queue, negative_queue], ignore_index=True)

for row in all_samples.itertuples(index=False):
    sample_rate, signal = wavfile.read(row.wav_path)
    # row.label: 1 为正样本，0 为负样本
```

当前检查到的音频数量：

| 类型 | 音频目录 | WAV 数量 |
| --- | --- | --- |
| 正样本 | `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate` | 18116 |
| 负样本 | `D:\Project_Github\Indo-Pacific-humpback-dolphin\13_Negative_Bandpass_20k_190k_NoHFER\Negative_Samples` | 90580 |
