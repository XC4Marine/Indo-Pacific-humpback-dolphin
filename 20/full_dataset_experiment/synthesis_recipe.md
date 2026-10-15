# 接收 click 信号合成指导

合成模型：`y[n] = A_click * sum_k a_k d[n - tau_k F_s] + r[n]`。

1. 从 `synthesis_pairs.csv` 按 `synthesis_pair_id` 整组选择 `atom_id` 与对应信道，不能分别随机抽取。
2. 直达径设为 `tau_0=0`，使用 `relative_signed_amplitude=a_k/|a_0|`；负值表示额外 π 相移。
3. 从 `atoms.npz` 读取真实原子波形及峰值对齐的复频谱，保留残余相位，不用中心频率生成理想脉冲。
4. 非整数时延使用频域相移或分数延迟滤波器，不能直接四舍五入。
5. 归一化信号最后统一设峰值或 RMS；物理声压必须另有水听器灵敏度、前置增益和 ADC 标定。
6. 从 `joint_residuals.npz` 选择相同 Voyage、相近 SNR 的真实拟合残差做 0.10 ms 块采样，不用白噪声替代。
7. 合成后复核中心频率、−3 dB 带宽、时延、相对幅值、相位、SNR 与削波情况。
8. 优先使用真实“原子—信道”配对；扩大样本时才按 `frequency_amplitude_phase_statistics.csv` 的条件联合分布采样。

当前高置信度非直达径 6785 条，其中严格高置信度 6611 条。本项目只提供流程和参数表，不实现合成函数，也不生成示例 WAV。
