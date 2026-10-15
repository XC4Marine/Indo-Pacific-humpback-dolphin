# 实验 v7：选择性多径物理置信度

三个入口相互隔离：

```powershell
python 20/experiment_v7/extract_physical_features_v7.py --ensemble 20 --output-root 20/experiment_v7/outputs
python 20/experiment_v7/train_evaluate_confidence_v7.py --feature-root 20/experiment_v7/outputs --output-root 20/experiment_v7/evaluation
python 20/experiment_v7/apply_pseudolabels_v7.py --model 20/experiment_v7/evaluation/physical_confidence_model.json --feature-root 20/experiment_v7/outputs --output-root 20/experiment_v7/pseudolabels
```

特征提取和部署入口不读取仿真真值；第二个入口是唯一真值读取点。仅当嵌套分组 OOF 结果满足预注册门槛，评估入口才会写入锁定模型，部署入口才会产生可接受伪标签。所有概率均限于当前模拟分布与可辨识性假设，不是无条件真实世界物理概率。
