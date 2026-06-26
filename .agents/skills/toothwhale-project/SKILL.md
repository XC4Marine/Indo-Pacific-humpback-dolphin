---
name: toothwhale-project
description: >
  Indo-Pacific humpback dolphin (Sousa chinensis) bioacoustics research project. Covers click detection,
  duration analysis, reverberation classification, data splitting, pre-trial analysis, and pulse train
  localization. All code runs under `conda activate toothwhale` (Python 3.11 at D:\Python_env\toothwhale\python.exe).
  Use when working on this project's code, data processing pipelines, MATLAB-to-Python migration, signal processing,
  or any task within the D:\Project_Github\Indo-Pacific-humpback-dolphin workspace.
---

# ToothWhale Project Skill

## 环境

- 工作目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin`
- Python：`D:\Python_env\toothwhale\python.exe` (Python 3.11.15)
- 激活：所有 .py 脚本通过 `& "D:\Python_env\toothwhale\python.exe" script.py` 直接执行
- 数据：`00_Data/ClickTrains/PulseTrain_001/` 至 `PulseTrain_832/`
- 已安装依赖：numpy, scipy, matplotlib, soundfile, pandas, tqdm

## 项目结构

参见 `.agents/project-knowledge.md`，每次任务完成后更新该文件。

## 代码规范

- 所有新 .py 文件添加 UTF-8 编码声明：`# -*- coding: utf-8 -*-`
- 第一行 shebang：`#!/usr/bin/env python3`
- 函数和类添加中文 docstring
- 关键计算步骤添加中文注释
- 变量名使用英文小写+下划线
- 路径使用 `pathlib.Path`，不用字符串拼接
- 所有文件输出路径使用绝对路径

## MATLAB 迁移注意事项

- `audioread` → `soundfile.read`
- `xcorr` → `scipy.signal.correlate` 或 `numpy.correlate`
- `hann` → `numpy.hanning` 或 `scipy.signal.windows.hann`
- `prctile` → `numpy.percentile`
- `movmean` → `numpy.convolve` 或 `pandas.Series.rolling.mean`
- MATLAB 1-based → Python 0-based 索引
- 注意 MATLAB 列优先 vs NumPy 行优先

## 输出规范

- 输出放在对应模块文件夹下
- 遍历多个输入时使用 tqdm 显示进度
- 每个独立 pulse train 处理失败时跳过并打印警告

## 沟通规范

- 称呼用户为 Chico
- 不确定时先询问，不随意假设
- 先探索代码再给出方案
- 代码修改后说明改了什么、为什么、如何验证

## 文档维护（重要）

### 实验记录
每次任务完成后，必须在 `.agents/experiment-log.md` 中追加一条记录，包含：
- 日期
- 完成了什么工作
- 关键指标与结果（如处理数量、准确率、耗时等）
- 生成了哪些代码和输出文件
- 与之前实验记录的关联（上游依赖了什么、下游谁会用到）

### 项目认知文档
`.agents/project-knowledge.md` 保存对项目的完整认知，每次完成任务后更新。内容包括：
- 目录结构与各模块功能
- 数据特点（格式、规模、关键参数）
- 项目整体目标
- 各模块代码的功能和分布
- 当前进度状态
- 用户的需求和偏好

### 验收要求
- 在计划阶段（Plan 模式），必须向用户明确确认验收指标：任务完成后要达到什么效果、什么量化指标
- 完成任务后，先按验收指标自主验证，成功后再向用户报告
- 如果验证不通过，告知用户遇到的困难，不假装成功

### .gitignore
- 不上传 `.png`、`.wav`、`.csv` 等大型数据文件
- 不上传 `00_Data/` 等数据目录
- 不上传各模块的输出子文件夹
- 不上传 `__pycache__/`、`.ipynb_checkpoints/` 等临时文件
- 只上传 .py、.ipynb、.md、.txt 等代码和文档
- .gitignore 文件已位于仓库根目录，内容是权威的