# 第22实验：测试 OOF 潜变量 CCA / Kernel CCA 诊断

每个折由独立训练的最佳检查点提取潜变量，并在折内完成分析；报告为五折均值 ± 标准差。
不同折的原始潜变量没有拼接，因为其坐标系由不同模型学习。

| 方法 | 样本口径 | 分析 | 第一典型相关 | 前10个均值 |
|---|---|---|---:|---:|
| Recon + BCE (without MIM) | 全体测试 | 线性 CCA | 0.9891 ± 0.0060 | 0.8628 ± 0.0141 |
| Recon + BCE (without MIM) | 全体测试 | Nyström RBF Kernel CCA | 0.9975 ± 0.0014 | 0.9788 ± 0.0048 |
| Recon + BCE (without MIM) | 真实正类 click | 线性 CCA | 0.9883 ± 0.0034 | 0.8710 ± 0.0190 |
| Recon + BCE (without MIM) | 真实正类 click | Nyström RBF Kernel CCA | 0.9981 ± 0.0007 | 0.9890 ± 0.0034 |
| Recon + BCE + positive-only MIM (weight 1.0) | 全体测试 | 线性 CCA | 0.9946 ± 0.0029 | 0.8805 ± 0.0239 |
| Recon + BCE + positive-only MIM (weight 1.0) | 全体测试 | Nyström RBF Kernel CCA | 0.9992 ± 0.0006 | 0.9868 ± 0.0019 |
| Recon + BCE + positive-only MIM (weight 1.0) | 真实正类 click | 线性 CCA | 0.9957 ± 0.0019 | 0.8882 ± 0.0307 |
| Recon + BCE + positive-only MIM (weight 1.0) | 真实正类 click | Nyström RBF Kernel CCA | 0.9996 ± 0.0002 | 0.9903 ± 0.0028 |
| Recon + BCE + cosine orthogonality (lambda 0.3) | 全体测试 | 线性 CCA | 0.9895 ± 0.0060 | 0.8646 ± 0.0097 |
| Recon + BCE + cosine orthogonality (lambda 0.3) | 全体测试 | Nyström RBF Kernel CCA | 0.9981 ± 0.0011 | 0.9812 ± 0.0026 |
| Recon + BCE + cosine orthogonality (lambda 0.3) | 真实正类 click | 线性 CCA | 0.9884 ± 0.0056 | 0.8705 ± 0.0158 |
| Recon + BCE + cosine orthogonality (lambda 0.3) | 真实正类 click | Nyström RBF Kernel CCA | 0.9984 ± 0.0004 | 0.9900 ± 0.0013 |

## 本次诊断的直接观察

在真实正类 click 上，三种方法的线性典型相关前10个均值依次为：无 MIM 0.8710、MIM=1.0 0.8882、正交 λ=0.3 0.8705。
对应的 Nyström RBF Kernel CCA 前10个均值依次为：0.9890、0.9903、0.9900。
因此，本次单种子五折机制诊断没有显示余弦正交将两组潜变量的可预测共享结构降至低于无 MIM 重建基线；MIM=1.0 的共享结构反而更高。

## 解释边界

典型相关高表示一个潜变量群中仍存在可由另一潜变量群线性或非线性预测的共享变化。
余弦正交只约束同一样本潜变量的夹角，因此不保证 CCA 或 Kernel CCA 必然较低。
该诊断不等同于统计独立性，也不证明真实物理信道已经恢复；Kernel CCA 使用 Nyström RBF 近似。
