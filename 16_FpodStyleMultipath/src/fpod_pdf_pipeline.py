"""实现与论文描述结构对应的 F-POD 风格点击检测和多途分析流程。

本模块负责把预处理后的水声信号重采样到 F-POD 工作采样率，提取类似 FP1
的逐点击记录，将相近点击组织成多途簇和点击串，并转换为统一的多途特征。
"""

# 推迟类型注解求值，允许在运行时尚未定义相关类型时使用其名称。
from __future__ import annotations

# 导入标准库 CSV 模块，用于把点击记录写入表格文件。
import csv
# 导入 dataclass 装饰器，用于声明只承载数据的不可变记录类型。
from dataclasses import dataclass
# 导入 Fraction，以精确计算多相重采样所需的整数升、降采样比例。
from fractions import Fraction
# 导入 Path，统一处理跨平台文件路径。
from pathlib import Path

# 导入 NumPy，执行数组运算、统计计算和数值类型转换。
import numpy as np
# 导入波峰检测与多相重采样函数。
from scipy.signal import find_peaks, resample_poly

# 导入项目统一的多途特征数据结构。
from .multipath_detection import MultipathFeature


# 把配置类声明为不可变数据类，防止分析过程中参数被意外修改。
@dataclass(frozen=True)
class FpodPdfSettings:
    """集中保存 F-POD 风格检测、点击串划分和多途聚类参数。"""

    target_sample_rate: int = 4_000_000  # F-POD 风格处理使用的目标采样率，单位为 Hz。
    min_frequency_khz: float = 17.0  # 可接受点击载频的下限，单位为 kHz。
    max_frequency_khz: float = 220.0  # 可接受点击载频的上限，单位为 kHz。
    amplitude_percentiles: tuple[float, ...] = (85.0, 90.0, 95.0, 97.0)  # 待比较的振幅阈值百分位。
    min_cycles: int = 2  # 构成有效点击所需的最少波形周期数。
    max_cycles: int = 255  # 单条记录能够保存的最大波形周期数。
    ipi_change_tolerance: float = 0.50  # 相邻脉冲间隔的最大相对变化比例。
    half_peak_drop_fraction: float = 0.50  # 确认波峰时要求谷值下降到峰值的比例。
    low_cycle_threshold_steps: tuple[float, ...] = (2.2, 1.8, 1.45, 1.2)  # 少周期点击使用的附加振幅倍数。
    train_min_clicks: int = 3  # 一个有效点击串至少包含的代表点击数。
    train_min_ici_us: float = 3_000.0  # 点击串内相邻点击间隔下限，单位为 μs。
    train_max_ici_us: float = 250_000.0  # 点击串内相邻点击间隔上限，单位为 μs。
    train_ici_change_tolerance: float = 0.75  # 点击串内相邻 ICI 的最大相对变化比例。
    split_window_us: float = 400.0  # 预留的点击拆分时间窗参数，单位为 μs。
    multipath_window_us: float = 3_000.0  # 同一多途簇允许的最大到达时间差，单位为 μs。
    multipath_frequency_tolerance_khz: float = 90.0  # 同一多途簇允许的最大频率差，单位为 kHz。


# 把点击记录声明为不可变数据类，确保记录生成后字段保持一致。
@dataclass(frozen=True)
class FpodClickRecord:
    """保存单个类似 FP1 点击及其点击串、多途簇归属信息。"""

    record_id: int  # 按时间顺序分配的点击记录编号。
    sample_index: int  # 最强波峰在 F-POD 风格信号中的样本索引。
    time_5us: int  # 点击到达时间的 5 μs 量化值。
    time_us: float  # 点击到达时间，单位为 μs。
    n_cycles: int  # 当前点击包含的有效波形周期数量。
    loudest_wave_number: int  # 最强波形在点击内部的序号，从 1 开始。
    ipi_range_250ns: int  # 点击内最大与最小 IPI 的差，量化单位为 250 ns。
    ipi_pk_minus1_250ns: int  # 最强波形之前一段 IPI，量化单位为 250 ns。
    ipi_pk_250ns: int  # 最强波形位置对应的 IPI，量化单位为 250 ns。
    ipi_pk_plus1_250ns: int  # 最强波形之后第一段 IPI，量化单位为 250 ns。
    ipi_pk_plus2_250ns: int  # 最强波形之后第二段 IPI，量化单位为 250 ns。
    amp_pk_minus1: int  # 最强波形前一波形的 8 位量化振幅。
    amp_pmax: int  # 点击内最强波形的 8 位量化振幅。
    amp_pk_plus1: int  # 最强波形后一波形的 8 位量化振幅。
    ipi_before_click_250ns: int  # 点击前 IPI；当前流程未计算，默认写入 0。
    amplitude_reversals: int  # 点击内部振幅变化方向反转的次数。
    duration_5us: int  # 点击持续时间的 5 μs 量化值。
    last_ipi_250ns: int  # 点击内最后一段 IPI，量化单位为 250 ns。
    sonar_flag: int  # 声呐标志位；当前流程未分类，默认写入 0。
    frequency_khz: float  # 根据 IPI 估算的点击中心频率，单位为 kHz。
    bandwidth_index: float  # 根据 IPI 与振幅离散程度计算的带宽代理指标。
    polarity: int  # 检出点击的极性，正极性为 1，负极性为 -1。
    train_id: int = 0  # 点击所属点击串编号；0 表示未归入有效点击串。
    multipath_cluster_id: int = 0  # 点击所属多途簇编号；0 表示单独点击。
    multipath_cluster_size: int = 1  # 当前多途簇包含的点击记录数量。
    cluster_low_frequency_khz: float = np.nan  # 当前多途簇的最低频率，单位为 kHz。
    cluster_high_frequency_khz: float = np.nan  # 当前多途簇的最高频率，单位为 kHz。


# 把管线返回值声明为不可变数据类，集中交付全部分析结果。
@dataclass(frozen=True)
class FpodPipelineResult:
    """汇总 F-POD 风格处理流程的记录、特征、参数和结构统计量。"""

    features: list[MultipathFeature]  # 转换后的统一多途特征列表。
    click_records: list[FpodClickRecord]  # 类似 FP1 的逐点击记录列表。
    sample_rate: int  # 流程内部实际使用的采样率，单位为 Hz。
    detection_choice: dict[str, float]  # 自动选中的阈值及相关检测参数。
    train_count: int  # 最终识别出的有效点击串数量。
    multipath_cluster_count: int  # 最终识别出的多途簇数量。


def run_fpod_pdf_pipeline(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings | None = None,
) -> FpodPipelineResult:
    """依次执行重采样、点击检测、多途聚类、点击串划分和特征转换。"""

    # 未显式传入配置时创建一套默认配置。
    settings = settings or FpodPdfSettings()
    # 把输入信号转换到 F-POD 风格目标采样率，并接收实际采样率与新信号。
    fpod_sample_rate, fpod_signal = resample_to_fpod_rate(signal, sample_rate, settings)
    # 在多个候选振幅阈值下提取点击，并保留评分最优的记录与阈值信息。
    records, choice = extract_fpod_click_records(fpod_signal, fpod_sample_rate, settings)
    # 先按时间和频率接近程度为点击记录分配多途簇。
    records = assign_multipath_clusters(records, settings)
    # 再从多途簇代表点击和单独点击中识别点击串。
    records = assign_trains(records, settings)
    # 把内部点击记录转换为项目其他模块通用的多途特征。
    features = records_to_multipath_features(records, fpod_signal, fpod_sample_rate)
    # 使用集合去重并统计大于 0 的有效点击串编号。
    train_count = len({item.train_id for item in records if item.train_id > 0})
    # 使用集合去重并统计大于 0 的有效多途簇编号。
    cluster_count = len({item.multipath_cluster_id for item in records if item.multipath_cluster_id > 0})
    # 将各阶段结果和统计量封装为统一返回对象。
    return FpodPipelineResult(
        features=features,  # 保存统一多途特征。
        click_records=records,  # 保存带聚类和点击串编号的点击记录。
        sample_rate=fpod_sample_rate,  # 保存内部处理采样率。
        detection_choice=choice,  # 保存最终采用的阈值选择。
        train_count=train_count,  # 保存点击串统计数量。
        multipath_cluster_count=cluster_count,  # 保存多途簇统计数量。
    )


def resample_to_fpod_rate(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings,
) -> tuple[int, np.ndarray]:
    """将输入信号重采样到配置的目标采样率，并归一化峰值振幅。"""

    # 采样率已满足目标值时直接返回，避免不必要的重采样和数值误差。
    if sample_rate == settings.target_sample_rate:
        # 返回原采样率和原信号对象。
        return sample_rate, signal
    # 把目标与原采样率之比约化为分母不超过 1000 的整数比例。
    ratio = Fraction(settings.target_sample_rate, sample_rate).limit_denominator(1000)
    # 使用约化比例执行抗混叠多相重采样。
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    # 计算重采样信号的绝对峰值；空数组的峰值定义为 0。
    peak = float(np.max(np.abs(resampled))) if len(resampled) else 0.0
    # 仅在峰值为正时执行归一化，避免除零。
    if peak > 0.0:
        # 将重采样信号缩放到绝对峰值为 1。
        resampled = resampled / peak
    # 返回目标采样率，并确保信号采用双精度浮点 NumPy 数组。
    return settings.target_sample_rate, np.asarray(resampled, dtype=np.float64)


def extract_fpod_click_records(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings,
) -> tuple[list[FpodClickRecord], dict[str, float]]:
    """比较多个振幅百分位阈值，并选择点击数量最接近目标规模的结果。"""

    # 对信号取绝对值得到整流振幅，供百分位阈值计算使用。
    rectified = np.abs(signal)
    # 初始化最佳候选；元组依次存放评分、记录列表和阈值信息。
    best: tuple[float, list[FpodClickRecord], dict[str, float]] | None = None
    # 依次测试配置中给出的每个振幅百分位。
    for percentile in settings.amplitude_percentiles:
        # 计算当前百分位对应的绝对振幅阈值。
        threshold = float(np.percentile(rectified, percentile))
        # 非正阈值无法区分有效波峰，因此跳过当前候选。
        if threshold <= 0.0:
            # 继续测试下一个百分位阈值。
            continue
        # 分别检测正负极性波峰，并构建当前阈值下的点击记录。
        records = _records_at_threshold(signal, sample_rate, threshold, settings)
        # 以记录数接近 120 为主、较高百分位略占优势的方式计算候选评分。
        score = abs(len(records) - 120) + (97.0 - percentile) * 0.2
        # 尚无最佳结果或当前评分更小时，用当前候选替换最佳结果。
        if best is None or score < best[0]:
            # 保存当前评分、记录列表和便于追踪的检测参数。
            best = (
                score,  # 候选评分，数值越小越优。
                records,  # 当前阈值提取的点击记录。
                {
                    "amplitude_percentile": percentile,  # 采用的振幅百分位。
                    "amplitude_threshold": threshold,  # 百分位对应的实际振幅阈值。
                    "click_records": float(len(records)),  # 当前阈值提取的记录数。
                    "ipi_change_tolerance": settings.ipi_change_tolerance,  # 波峰分组使用的 IPI 容差。
                },
            )
    # 所有候选阈值均无效时返回空记录和显式的缺失参数。
    if best is None:
        # 构造与正常返回值字段一致的空检测结果。
        return [], {
            "amplitude_percentile": np.nan,  # 无有效百分位时记为 NaN。
            "amplitude_threshold": np.nan,  # 无有效阈值时记为 NaN。
            "click_records": 0.0,  # 空结果的记录数为 0。
            "ipi_change_tolerance": settings.ipi_change_tolerance,  # 仍记录本次使用的 IPI 容差。
        }
    # 丢弃最佳候选的评分，仅保留记录和参数字典。
    _, records, choice = best
    # 返回最佳阈值下的点击记录和阈值选择信息。
    return records, choice


def _records_at_threshold(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> list[FpodClickRecord]:
    """在给定振幅阈值下检测正负极性点击，并抑制时间上重复的记录。"""

    # 创建列表，收集正、负两个极性方向生成的全部候选记录。
    all_records: list[FpodClickRecord] = []
    # 原信号用于检测正极性波峰，取反信号用于以相同算法检测负极性波峰。
    for polarity, oriented in ((1, signal), (-1, -signal)):
        # 检测当前极性方向中满足阈值、距离和峰谷下降条件的波峰。
        peaks = _confirmed_wave_peaks(oriented, sample_rate, threshold, settings)
        # 根据频率范围及 IPI 变化连续性把相邻波峰组织为候选点击。
        groups = _group_wave_peaks(peaks, sample_rate, settings)
        # 逐个候选波峰组尝试构建 F-POD 点击记录。
        for group in groups:
            # 使用当前已收集记录数生成临时连续编号，并传入原始信号提取振幅。
            record = _make_click_record(
                len(all_records) + 1,  # 当前候选记录的临时编号。
                group,  # 构成候选点击的波峰索引。
                signal,  # 保留真实正负振幅的原始信号。
                sample_rate,  # 信号采样率，单位为 Hz。
                polarity,  # 当前检测方向对应的极性标志。
                threshold,  # 当前百分位对应的振幅阈值。
                settings,  # F-POD 检测配置。
            )
            # 只有通过周期数、振幅与频率检查的记录才加入候选列表。
            if record is not None:
                # 保存当前有效候选记录。
                all_records.append(record)

    # 按最强波峰的样本位置排序，使正负极性记录进入统一时间顺序。
    all_records.sort(key=lambda item: item.sample_index)
    # 创建去重后的记录列表。
    suppressed: list[FpodClickRecord] = []
    # 把 5 μs 的重复抑制窗口换算为当前采样率下的样本数。
    suppress_samples = int(round(5.0e-6 * sample_rate))
    # 按时间顺序检查每一条候选记录。
    for record in all_records:
        # 当前记录与最近保留记录相距不超过抑制窗口时，视为同一点击。
        if suppressed and record.sample_index - suppressed[-1].sample_index <= suppress_samples:
            # 重复候选中仅保留量化峰值振幅更大的记录。
            if record.amp_pmax > suppressed[-1].amp_pmax:
                # 用更强的当前记录替换最近保留记录。
                suppressed[-1] = record
            # 无论是否替换，当前重复记录均不再追加。
            continue
        # 当前记录不与前一条冲突时，将其作为新点击保留。
        suppressed.append(record)

    # 去重可能打乱临时编号，因此按最终时间顺序重新连续编号。
    return [
        _renumber_record(record, i + 1)  # 将当前记录编号改为从 1 开始的最终序号。
        for i, record in enumerate(suppressed)
    ]


def _confirmed_wave_peaks(
    oriented: np.ndarray,
    sample_rate: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> list[int]:
    """检测超过阈值且峰后振幅充分下降的波峰样本索引。"""

    # 根据最高允许频率估算相邻物理波峰的最小样本距离，并确保至少为 1。
    min_distance = max(1, int(round((1_000.0 / settings.max_frequency_khz) * 1e-6 * sample_rate)))
    # 在已统一极性的信号中检测超过阈值且满足最小距离的波峰。
    peaks, props = find_peaks(oriented, height=threshold, distance=min_distance)
    # 创建列表保存通过峰后谷值检查的确认波峰。
    confirmed: list[int] = []
    # 把 NumPy 整数索引转为 Python 整数，并同时保留循环位置。
    for i, peak in enumerate(map(int, peaks)):
        # 使用下一波峰作为搜索终点；末峰则向后检查最多三个最小峰距。
        next_peak = int(peaks[i + 1]) if i + 1 < len(peaks) else min(len(oriented), peak + min_distance * 3)
        # 计算当前峰与下一峰之间的最低振幅；空区间时退回当前峰值。
        valley = float(np.min(oriented[peak:next_peak])) if next_peak > peak else float(oriented[peak])
        # 仅当峰后谷值下降到设定峰值比例以下时确认该波峰。
        if valley <= oriented[peak] * settings.half_peak_drop_fraction:
            # 保存通过检查的波峰样本索引。
            confirmed.append(peak)
    # 返回按时间顺序排列的确认波峰索引。
    return confirmed


def _group_wave_peaks(
    peaks: list[int],
    sample_rate: int,
    settings: FpodPdfSettings,
) -> list[list[int]]:
    """依据瞬时频率范围和相邻 IPI 稳定性将波峰划分为点击组。"""

    # 没有波峰时直接返回空组列表。
    if not peaks:
        # 返回空结果，避免访问不存在的第一个波峰。
        return []
    # 创建最终波峰组列表。
    groups: list[list[int]] = []
    # 使用第一个波峰初始化当前候选点击。
    current = [peaks[0]]
    # 当前组尚未产生 IPI，因此先把前一 IPI 记为空。
    previous_ipi_us: float | None = None
    # 成对遍历相邻波峰，判断后一波峰是否应并入当前点击。
    for previous, peak in zip(peaks[:-1], peaks[1:]):
        # 把相邻波峰的样本差换算为微秒单位的脉冲间隔。
        ipi_us = (peak - previous) / sample_rate * 1e6
        # 根据 IPI 的倒数估算瞬时频率；非正 IPI 记为 NaN。
        frequency_khz = 1000.0 / ipi_us if ipi_us > 0.0 else np.nan
        # 检查瞬时频率是否落入配置允许的点击频带。
        valid_frequency = settings.min_frequency_khz <= frequency_khz <= settings.max_frequency_khz
        # 检查当前 IPI 与前一 IPI 的相对变化是否在容差内。
        valid_change = (
            previous_ipi_us is None  # 当前组的第一段 IPI 无需比较变化。
            or abs(ipi_us - previous_ipi_us) / max(previous_ipi_us, 1e-12)
            <= settings.ipi_change_tolerance
        )
        # 频率有效且 IPI 变化连续时，把新波峰并入当前组。
        if valid_frequency and valid_change:
            # 将当前波峰追加到正在构建的点击组。
            current.append(peak)
            # 保存当前 IPI，供下一对波峰进行变化率比较。
            previous_ipi_us = ipi_us
        # 任一条件不满足时结束当前组，并从当前波峰开始新组。
        else:
            # 把已经完成的当前波峰组加入结果列表。
            groups.append(current)
            # 使用导致断开的当前波峰初始化下一组。
            current = [peak]
            # 新组尚未形成 IPI，因此重置前一 IPI。
            previous_ipi_us = None
    # 循环结束后保存尚未加入结果的最后一个波峰组。
    groups.append(current)
    # 返回按时间顺序划分的全部候选点击组。
    return groups


def _make_click_record(
    record_id: int,
    group: list[int],
    signal: np.ndarray,
    sample_rate: int,
    polarity: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> FpodClickRecord | None:
    """校验一个波峰组并将其量化为单条类似 FP1 的点击记录。"""

    # 波峰数量等价于当前候选点击包含的周期数。
    n_cycles = len(group)
    # 周期数超出 F-POD 记录允许范围时拒绝当前候选。
    if not settings.min_cycles <= n_cycles <= settings.max_cycles:
        # 返回空值表示当前波峰组不能构成有效点击。
        return None
    # 从原始信号提取各波峰的绝对振幅。
    amplitudes = np.abs(signal[group])
    # 查找绝对振幅最大波峰在当前组内的零起始位置。
    loudest_idx = int(np.argmax(amplitudes))
    # 根据周期数选择少周期点击所需的额外阈值倍数。
    low_cycle_multiplier = _threshold_multiplier(n_cycles, settings)
    # 最强波峰仍低于修正阈值时拒绝当前候选。
    if float(amplitudes[loudest_idx]) < threshold * low_cycle_multiplier:
        # 返回空值，表示振幅不足以支持当前周期数的点击。
        return None

    # 计算相邻波峰之间的 IPI，并转换为微秒。
    ipi_us = np.diff(group) / sample_rate * 1e6
    # 没有任何 IPI 时无法估算点击频率。
    if len(ipi_us) == 0:
        # 返回空值，表示波峰组缺少频率估算所需信息。
        return None
    # 把 IPI 除以 0.25 μs 并四舍五入，得到 250 ns 量化整数。
    ipi_250ns = np.rint(ipi_us / 0.25).astype(int)
    # 根据周期数量和最强波峰位置估算 F-POD 风格中心频率。
    frequency_khz = _fpod_frequency_khz(ipi_us, n_cycles, loudest_idx)
    # 频率为 NaN 或无穷值时拒绝当前候选。
    if not np.isfinite(frequency_khz):
        # 返回空值，表示无法得到有效中心频率。
        return None

    # 取最强波峰的绝对样本索引作为点击时间参考点。
    pmax_sample = group[loudest_idx]
    # 使用首尾波峰样本差计算点击持续时间，单位为 μs。
    duration_us = (group[-1] - group[0]) / sample_rate * 1e6
    # 将归一化绝对振幅缩放、四舍五入并限制到 0—255 的 8 位整数范围。
    amp_u8 = np.clip(np.rint(amplitudes * 255.0), 0, 255).astype(int)
    # 用计算和量化后的全部字段构造不可变点击记录。
    return FpodClickRecord(
        record_id=record_id,  # 保存当前点击的临时或最终编号。
        sample_index=pmax_sample,  # 保存最强波峰样本位置。
        time_5us=int(round((pmax_sample / sample_rate * 1e6) / 5.0)),  # 保存 5 μs 量化到达时间。
        time_us=pmax_sample / sample_rate * 1e6,  # 保存浮点微秒到达时间。
        n_cycles=n_cycles,  # 保存有效周期数。
        loudest_wave_number=loudest_idx + 1,  # 将零起始索引转换为从 1 开始的波形序号。
        ipi_range_250ns=int(np.max(ipi_250ns) - np.min(ipi_250ns)) if len(ipi_250ns) else 0,  # 保存 IPI 极差。
        ipi_pk_minus1_250ns=_ipi_at(ipi_250ns, loudest_idx - 1),  # 保存峰前 IPI。
        ipi_pk_250ns=_ipi_at(ipi_250ns, loudest_idx),  # 保存峰位 IPI。
        ipi_pk_plus1_250ns=_ipi_at(ipi_250ns, loudest_idx + 1),  # 保存峰后第一段 IPI。
        ipi_pk_plus2_250ns=_ipi_at(ipi_250ns, loudest_idx + 2),  # 保存峰后第二段 IPI。
        amp_pk_minus1=_amp_at(amp_u8, loudest_idx - 1),  # 保存最强峰前一波形振幅。
        amp_pmax=int(amp_u8[loudest_idx]),  # 保存最强波形振幅。
        amp_pk_plus1=_amp_at(amp_u8, loudest_idx + 1),  # 保存最强峰后一波形振幅。
        ipi_before_click_250ns=0,  # 当前算法未计算点击前 IPI，因此写入 0。
        amplitude_reversals=_amplitude_reversals(amp_u8),  # 保存振幅方向反转次数。
        duration_5us=int(round(duration_us / 5.0)),  # 保存 5 μs 量化点击时长。
        last_ipi_250ns=int(ipi_250ns[-1]),  # 保存最后一段 IPI。
        sonar_flag=0,  # 当前算法未执行声呐分类，因此写入 0。
        frequency_khz=frequency_khz,  # 保存估算中心频率。
        bandwidth_index=_bandwidth_index(ipi_250ns, amp_u8, loudest_idx),  # 保存带宽代理指标。
        polarity=polarity,  # 保存检出波峰的极性。
    )


def _threshold_multiplier(n_cycles: int, settings: FpodPdfSettings) -> float:
    """返回少周期点击所需的振幅阈值修正倍数。"""

    # 五个及以上周期已具备较强连续性，不再提高基础阈值。
    if n_cycles >= 5:
        # 返回不改变基础阈值的单位倍数。
        return 1.0
    # 将 2—4 周期映射到配置中的阶梯倍数，并把索引限制在合法范围。
    return settings.low_cycle_threshold_steps[max(0, min(3, 4 - n_cycles))]


def _fpod_frequency_khz(ipi_us: np.ndarray, n_cycles: int, loudest_idx: int) -> float:
    """按周期数量选择局部中位 IPI 或累计时长估算点击频率。"""

    # 四个及以下周期使用最强波峰附近的局部 IPI，以减少边缘影响。
    if n_cycles <= 4:
        # 收集最强波峰前后至多四段处于数组范围内的 IPI。
        near = [
            ipi_us[i]  # 读取当前合法位置的 IPI。
            for i in (loudest_idx - 1, loudest_idx, loudest_idx + 1, loudest_idx + 2)
            if 0 <= i < len(ipi_us)
        ]
        # 使用局部 IPI 中位数的倒数换算频率；没有合法 IPI 时返回 NaN。
        return float(1000.0 / np.median(near)) if near else np.nan
    # 较长点击最多累计前 16 段 IPI 的持续时间。
    duration_us = float(np.sum(ipi_us[: min(16, len(ipi_us))]))
    # 记录实际参与累计的 IPI 段数，最多为 16。
    cycles = min(16, len(ipi_us))
    # 用周期数除以累计时长换算 kHz；非正时长返回 NaN。
    return float(1000.0 * cycles / duration_us) if duration_us > 0.0 else np.nan


def _ipi_at(values: np.ndarray, index: int) -> int:
    """安全读取指定位置的 IPI，越界时返回 F-POD 占位值 0。"""

    # 索引有效时返回对应整数，否则返回 0。
    return int(values[index]) if 0 <= index < len(values) else 0


def _amp_at(values: np.ndarray, index: int) -> int:
    """安全读取指定位置的量化振幅，越界时返回占位值 0。"""

    # 索引有效时返回对应整数，否则返回 0。
    return int(values[index]) if 0 <= index < len(values) else 0


def _amplitude_reversals(amplitudes: np.ndarray) -> int:
    """统计点击内部相邻振幅差的符号反转次数。"""

    # 少于三个振幅无法形成两段差值，因此不存在可统计的反转。
    if len(amplitudes) < 3:
        # 返回 0 次反转。
        return 0
    # 转为浮点数后计算相邻波形振幅差，避免无符号整数减法溢出。
    diffs = np.diff(amplitudes.astype(float))
    # 删除为零的差值，再提取其正负符号。
    signs = np.sign(diffs[diffs != 0])
    # 少于两个非零变化方向时无法形成方向反转。
    if len(signs) < 2:
        # 返回 0 次反转。
        return 0
    # 比较相邻变化方向并统计符号不同的位置数量。
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _bandwidth_index(ipi_250ns: np.ndarray, amplitudes: np.ndarray, loudest_idx: int) -> float:
    """用 IPI 与振幅相对离散程度构造点击带宽代理指标。"""

    # 少于两段 IPI 时无法可靠计算离散程度。
    if len(ipi_250ns) < 2:
        # 返回较大的约定占位值，表示带宽指标不可用。
        return 999.0
    # 取最强波峰附近的 IPI 作为中心，并以 1 为下限避免除零。
    center_ipi = max(float(_ipi_at(ipi_250ns, min(loudest_idx, len(ipi_250ns) - 1))), 1.0)
    # 取最强波形振幅作为中心，并以 1 为下限避免除零。
    center_amp = max(float(amplitudes[loudest_idx]), 1.0)
    # 计算所有 IPI 相对中心 IPI 的归一化绝对偏差之和。
    ipi_change = float(np.sum(np.abs(ipi_250ns - center_ipi) / center_ipi))
    # 计算所有振幅相对中心振幅的归一化绝对偏差之和。
    amp_change = float(np.sum(np.abs(amplitudes - center_amp) / center_amp))
    # 将时序和振幅两类离散程度相加，得到最终带宽代理指标。
    return ipi_change + amp_change


def _renumber_record(record: FpodClickRecord, record_id: int) -> FpodClickRecord:
    """复制不可变点击记录，并仅替换其记录编号。"""

    # 展开原字段字典，用新编号覆盖 record_id 后构造新记录。
    return FpodClickRecord(**{**record.__dict__, "record_id": record_id})


def assign_multipath_clusters(
    records: list[FpodClickRecord],
    settings: FpodPdfSettings,
) -> list[FpodClickRecord]:
    """按到达时间窗和频率容差将点击记录划分为多途簇。"""

    # 创建列表保存更新了多途簇字段的记录。
    output: list[FpodClickRecord] = []
    # 使用记录编号集合追踪已归入某个候选簇的点击。
    used: set[int] = set()
    # 从 0 开始累计有效多途簇编号。
    cluster_id = 0
    # 按输入时间顺序把每条尚未使用的记录作为候选直接路径。
    for record in records:
        # 已经被更早候选记录吸收的点击无需重复建簇。
        if record.record_id in used:
            # 跳到下一条输入记录。
            continue
        # 收集尚未使用且同时满足时间差与频率差条件的候选成员。
        members = [
            candidate  # 保留当前满足全部条件的候选点击。
            for candidate in records
            if candidate.record_id not in used  # 排除已分配给早期簇的记录。
            and 0.0 <= candidate.time_us - record.time_us <= settings.multipath_window_us  # 限制非负到达延迟。
            and abs(candidate.frequency_khz - record.frequency_khz)
            <= settings.multipath_frequency_tolerance_khz  # 限制与候选直接路径的频率差。
        ]
        # 至少两个成员才构成有效多途簇。
        if len(members) > 1:
            # 为新发现的有效多途簇分配下一个正整数编号。
            cluster_id += 1
        # 单成员组使用 0 表示独立点击，多成员组使用新簇编号。
        current_cluster = cluster_id if len(members) > 1 else 0
        # 计算当前成员集合中的最低中心频率。
        low = min(item.frequency_khz for item in members)
        # 计算当前成员集合中的最高中心频率。
        high = max(item.frequency_khz for item in members)
        # 逐条复制成员，并写入一致的多途簇统计字段。
        for member in members:
            # 将当前成员编号加入已使用集合，防止被后续簇重复吸收。
            used.add(member.record_id)
            # 向输出列表追加更新后的不可变记录副本。
            output.append(
                # 通过字段字典展开复制原记录，仅覆盖多途簇相关字段。
                FpodClickRecord(
                    **{
                        **member.__dict__,  # 保留成员原有的全部点击字段。
                        "multipath_cluster_id": current_cluster,  # 写入当前多途簇编号。
                        "multipath_cluster_size": len(members),  # 写入当前簇成员数量。
                        "cluster_low_frequency_khz": low,  # 写入当前簇最低频率。
                        "cluster_high_frequency_khz": high,  # 写入当前簇最高频率。
                    }
                )
            )
    # 按样本索引恢复全局时间顺序并返回更新后的记录。
    return sorted(output, key=lambda item: item.sample_index)


def assign_trains(records: list[FpodClickRecord], settings: FpodPdfSettings) -> list[FpodClickRecord]:
    """从单独点击和各多途簇的首个成员中识别稳定的点击串。"""

    # 每个多途簇仅取最早成员作为代表，避免回声被误判为新的主动点击。
    representatives = [
        item  # 保留满足代表条件的当前记录。
        for item in records
        if item.multipath_cluster_id == 0  # 单独点击天然作为代表。
        or _is_cluster_representative(item, records)
    ]
    # 创建列表保存满足最小点击数要求的点击串。
    trains: list[list[FpodClickRecord]] = []
    # 创建当前正在延伸的候选点击串。
    current: list[FpodClickRecord] = []
    # 当前串尚无相邻点击间隔，因此前一 ICI 初始化为空。
    previous_ici_us: float | None = None
    # 按时间顺序逐个检查代表点击。
    for record in representatives:
        # 当前串为空时使用当前记录初始化新串。
        if not current:
            # 把当前记录设为候选串的第一个点击。
            current = [record]
            # 新串尚无 ICI，因此保持空值。
            previous_ici_us = None
            # 完成初始化后处理下一条代表记录。
            continue
        # 计算当前记录与候选串末记录之间的点击间隔，单位为 μs。
        ici_us = record.time_us - current[-1].time_us
        # 检查 ICI 是否处于配置允许的点击串时间范围。
        valid_ici = settings.train_min_ici_us <= ici_us <= settings.train_max_ici_us
        # 检查当前 ICI 相对前一 ICI 的变化是否处于容差范围。
        valid_change = (
            previous_ici_us is None  # 当前串第一段 ICI 不需要变化率比较。
            or abs(ici_us - previous_ici_us) / max(previous_ici_us, 1e-12)
            <= settings.train_ici_change_tolerance
        )
        # ICI 范围和变化率均有效时延伸当前候选串。
        if valid_ici and valid_change:
            # 把当前代表点击追加到候选串。
            current.append(record)
            # 保存当前 ICI，供下一个代表点击进行稳定性比较。
            previous_ici_us = ici_us
        # 任一条件无效时结束当前候选串，并从当前记录新建候选串。
        else:
            # 仅保存达到最小点击数要求的已结束候选串。
            if len(current) >= settings.train_min_clicks:
                # 把有效候选串加入最终点击串列表。
                trains.append(current)
            # 以当前记录作为下一候选串的起点。
            current = [record]
            # 新候选串尚无 ICI，因此重置为空。
            previous_ici_us = None
    # 循环结束后检查最后一个尚未保存的候选串。
    if len(current) >= settings.train_min_clicks:
        # 最后候选串有效时加入点击串列表。
        trains.append(current)

    # 创建“簇编号或负记录编号”到点击串编号的映射。
    train_by_cluster_or_record: dict[int, int] = {}
    # 从 1 开始为每个有效点击串分配编号。
    for train_id, train in enumerate(trains, start=1):
        # 逐个登记当前点击串中的代表记录。
        for item in train:
            # 多途成员以正簇编号共享键，单独点击以负记录编号获得唯一键。
            key = item.multipath_cluster_id if item.multipath_cluster_id > 0 else -item.record_id
            # 将当前键映射到所属点击串编号。
            train_by_cluster_or_record[key] = train_id

    # 创建列表保存写入点击串编号后的全部记录。
    assigned: list[FpodClickRecord] = []
    # 遍历原记录，使多途簇所有成员继承其代表记录的点击串编号。
    for record in records:
        # 按与代表记录相同的规则构造查询键。
        key = record.multipath_cluster_id if record.multipath_cluster_id > 0 else -record.record_id
        # 复制记录并写入映射到的点击串编号；无映射时使用 0。
        assigned.append(FpodClickRecord(**{**record.__dict__, "train_id": train_by_cluster_or_record.get(key, 0)}))
    # 返回保持原时间顺序的已分配记录列表。
    return assigned


def _is_cluster_representative(record: FpodClickRecord, records: list[FpodClickRecord]) -> bool:
    """判断记录是否为其多途簇中最早到达的代表点击。"""

    # 不属于多途簇的单独点击直接视为代表。
    if record.multipath_cluster_id == 0:
        # 返回真，允许单独点击参与点击串识别。
        return True
    # 查找与当前记录同簇成员中的最小样本索引。
    first = min(
        item.sample_index  # 读取同簇成员的到达样本位置。
        for item in records
        if item.multipath_cluster_id == record.multipath_cluster_id
    )
    # 仅最早到达的成员被视为当前多途簇代表。
    return record.sample_index == first


def records_to_multipath_features(
    records: list[FpodClickRecord],
    signal: np.ndarray,
    sample_rate: int,
) -> list[MultipathFeature]:
    """把内部 F-POD 点击记录转换为统一的多途特征列表。"""

    # 创建列表保存最终转换得到的通用多途特征。
    features: list[MultipathFeature] = []
    # 创建按正多途簇编号组织成员的字典。
    by_cluster: dict[int, list[FpodClickRecord]] = {}
    # 创建列表单独保存不属于多途簇的点击。
    singles: list[FpodClickRecord] = []
    # 遍历全部记录，并按是否属于多途簇进行分流。
    for record in records:
        # 正多途簇编号表示当前记录属于一个包含多个到达路径的事件。
        if record.multipath_cluster_id > 0:
            # 确保簇成员列表存在，并把当前记录追加到对应簇。
            by_cluster.setdefault(record.multipath_cluster_id, []).append(record)
        # 编号为 0 的记录作为单独点击处理。
        else:
            # 把当前单独点击加入 singles 列表。
            singles.append(record)

    # 逐个转换没有多途成员的单独点击。
    for record in singles:
        # 单独点击同时作为自身的直接路径，并使用 single 角色。
        feature = _record_to_feature(record, record, signal, sample_rate, "single")
        # 对单独点击沿用其点击串编号作为可视化聚类编号。
        feature.cluster_id = record.train_id
        # 单独点击不存在更早到达成员，因此标记为直接路径。
        feature.is_direct_path = True
        # 保存转换完成的单独点击特征。
        features.append(feature)

    # 按多途簇编号升序遍历每组成员。
    for cluster_id, members in sorted(by_cluster.items()):
        # 按样本索引排列成员，确保最早到达记录位于首位。
        members = sorted(members, key=lambda item: item.sample_index)
        # 将最早到达成员定义为直接路径参考。
        direct = members[0]
        # 同时获取成员排序位置和记录，用于分配路径角色。
        for rank, member in enumerate(members):
            # 首成员命名为 direct，其余成员按到达顺序命名为 multipath_1 等。
            role = "direct" if rank == 0 else f"multipath_{rank}"
            # 以最早到达成员为延迟与相对振幅参考，转换当前成员。
            feature = _record_to_feature(member, direct, signal, sample_rate, role)
            # 使用当前多途簇编号覆盖通用特征的聚类编号。
            feature.cluster_id = cluster_id
            # 仅排序第一的成员标记为直接路径。
            feature.is_direct_path = rank == 0
            # 保存转换完成的当前多途成员特征。
            features.append(feature)
    # 按实际到达时间排序，使单独点击和多途成员恢复统一时间顺序。
    return sorted(features, key=lambda item: item.arrival_time_us)


def _record_to_feature(
    record: FpodClickRecord,
    direct: FpodClickRecord,
    signal: np.ndarray,
    sample_rate: int,
    role: str,
) -> MultipathFeature:
    """将一条点击记录转换为相对指定直接路径的多途特征。"""

    # 把 8 位量化峰值恢复为 0—1 线性振幅，并设置极小下限避免对数为负无穷。
    amp = max(record.amp_pmax / 255.0, 1e-12)
    # 以相同方式计算直接路径参考振幅。
    direct_amp = max(direct.amp_pmax / 255.0, 1e-12)
    # 构造项目通用的多途特征对象。
    return MultipathFeature(
        click_id=direct.record_id,  # 使用直接路径记录编号标识同一物理点击事件。
        event_role=role,  # 保存 direct、single 或 multipath_n 路径角色。
        direct_sample=direct.sample_index,  # 保存直接路径样本索引作为时间参考。
        sample_index=record.sample_index,  # 保存当前路径的样本索引。
        arrival_time_us=record.time_us,  # 保存当前路径相对信号起点的到达时间。
        delay_us=record.time_us - direct.time_us,  # 保存当前路径相对直接路径的延迟。
        frequency_khz=record.frequency_khz,  # 保存当前点击记录的中心频率。
        amplitude_db=float(20.0 * np.log10(amp / direct_amp + 1e-12)),  # 计算相对直接路径振幅级。
        linear_amplitude=amp,  # 保存归一化线性峰值振幅。
        rms_energy=_local_energy(signal, sample_rate, record.sample_index),  # 计算点击附近 RMS 能量。
        n_wave_peaks=record.n_cycles,  # 使用有效周期数表示波峰数量。
        dominant_ipi_us=_dominant_ipi_us(record),  # 计算最强波峰附近的代表 IPI。
        click_duration_us=record.duration_5us * 5.0,  # 将量化时长恢复为微秒。
        frequency_points_khz=[record.frequency_khz],  # 用单个中心频率初始化频率点列表。
        frequency_strength=[0.0],  # 当前流程无频谱强度序列，因此写入零占位值。
    )


def _dominant_ipi_us(record: FpodClickRecord) -> float:
    """计算最强波峰附近有效 IPI 的中位数，单位为 μs。"""

    # 收集记录中围绕最强波峰保存的四个 250 ns 量化 IPI 字段。
    values = [
        record.ipi_pk_minus1_250ns,  # 最强波峰前一段 IPI。
        record.ipi_pk_250ns,  # 最强波峰位置对应 IPI。
        record.ipi_pk_plus1_250ns,  # 最强波峰后第一段 IPI。
        record.ipi_pk_plus2_250ns,  # 最强波峰后第二段 IPI。
    ]
    # 过滤掉由数组越界产生的零占位值。
    valid = [item for item in values if item > 0]
    # 有效值取中位数并乘 0.25 恢复微秒；没有有效值时返回 NaN。
    return float(np.median(valid) * 0.25) if valid else np.nan


def _local_energy(signal: np.ndarray, sample_rate: int, center_sample: int, half_us: float = 50.0) -> float:
    """计算点击中心前后指定半窗长度内信号的均方根能量。"""

    # 把微秒半窗转换为样本数，并保证窗口半宽至少为一个样本。
    half = max(1, int(round(half_us * 1e-6 * sample_rate)))
    # 计算窗口起点，并限制其不小于信号首样本。
    start = max(0, center_sample - half)
    # 计算包含中心及右端点的窗口终点，并限制其不超过信号长度。
    end = min(len(signal), center_sample + half + 1)
    # 从信号中截取当前点击附近的局部片段。
    segment = signal[start:end]
    # 极端边界情况下局部片段可能为空。
    if len(segment) == 0:
        # 空片段没有能量，返回 0。
        return 0.0
    # 计算片段平方均值的平方根，并转换为 Python 浮点数。
    return float(np.sqrt(np.mean(segment * segment)))


def write_fpod_records_csv(records: list[FpodClickRecord], output_path: Path) -> None:
    """把点击记录的全部数据类字段按声明顺序写入 UTF-8 CSV。"""

    # 创建输出文件的父目录；所需父级一并创建，已存在时不报错。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 从数据类元数据读取字段名称，并保持类中的声明顺序。
    fields = list(FpodClickRecord.__dataclass_fields__.keys())
    # 以文本写模式打开输出文件，并禁用额外换行以符合 csv 模块要求。
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        # 创建按字段名写出字典行的 CSV 写入器。
        writer = csv.DictWriter(handle, fieldnames=fields)
        # 写入一行由数据类字段名组成的表头。
        writer.writeheader()
        # 按输入顺序逐条遍历点击记录。
        for record in records:
            # 将数据类字段字典作为一行写入 CSV。
            writer.writerow(record.__dict__)
