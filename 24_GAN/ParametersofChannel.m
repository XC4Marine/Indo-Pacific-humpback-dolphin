% 使用 OMP 提取多径信道冲激响应并计算全维度物理/统计特征
clear; clc; close all;

%% 1. 导入数据
% ------------------- 替换为你的实际数据 -------------------
[s, fs] = audioread('FirstArrival_Train094_pulse4.wav'); % 读取源信号 (直达波/模板)
[y, fs_0] = audioread('Train094_Pulse_004.wav');         % 读取带多径的源信号

y = y(:, 1); % 取单声道
s = s(:, 1);
% --------------------------------------------------------

%% 2. 参数设置与字典矩阵构造
N = length(y); % 设置需要估计的最大信道长度
L = length(s);

% 确保矩阵维度匹配，理论卷积长度为 L + N - 1
if length(y) < L + N - 1
    y = [y; zeros(L + N - 1 - length(y), 1)]; % 补零
else
    y = y(1 : L + N - 1); % 截断
end

% 构造卷积字典矩阵 Phi
Phi = zeros(length(y), N);
for i = 1:N
    Phi(i : i+L-1, i) = s;
end

% 列归一化
norm_Phi = sqrt(sum(Phi.^2, 1));
norm_Phi(norm_Phi == 0) = 1; 
Phi_normalized = Phi ./ norm_Phi;

%% 3. OMP 算法核心实现 (固定 K = 15)
K = 15; 
residual = y; 
support = []; 

fprintf('开始 OMP 迭代计算 (目标多径数 K = %d)...\n', K);

for iter = 1:K
    correlations = Phi_normalized' * residual;
    [~, best_idx] = max(abs(correlations));
    support = [support, best_idx];
    
    Phi_sub = Phi(:, support);
    h_sub = Phi_sub \ y;
    residual = y - Phi_sub * h_sub;
end

% 构建最终的信道估计向量
h_est = zeros(N, 1);
h_est(support) = h_sub;
y_est = Phi * h_est;

% 计算信号相关系数
R = corrcoef(y, y_est);
corr_val = R(1, 2);

%% 4. 核心特征参数化计算
% 提取支撑集对应的延迟（采样点）、幅度、相位、功率
[sorted_support, sort_idx] = sort(support);
sorted_h_sub = h_sub(sort_idx);

tau_samples = sorted_support(:);              % 延迟采样点
tau_seconds = (tau_samples - 1) / fs;         % 绝对时延 (秒)
amplitudes = abs(sorted_h_sub(:));            % 衰减幅度 A_i
phases = angle(sorted_h_sub(:));              % 相位跳变 \theta_i (弧度)
powers = amplitudes.^2;                       % 功率时延谱 P(tau) = |h|^2

% --- (1) 时延扩展特征 (Time Dispersion Parameters) ---
total_power = sum(powers);
mean_excess_delay = sum(tau_seconds .* powers) / total_power; % 平均附加时延 (\bar{\tau})
rms_delay_spread = sqrt(sum((tau_seconds - mean_excess_delay).^2 .* powers) / total_power); % 均方根时延扩展 (\sigma_\tau)
max_excess_delay = max(tau_seconds) - min(tau_seconds); % 最大附加时延

% --- (2) 能量与衰落特征 (Energy & Fading Parameters) ---
total_gain = total_power; % 信道总增益
[max_path_power, max_idx] = max(powers);
scatter_power = total_power - max_path_power;
if scatter_power <= 0
    rician_K_dB = Inf;
else
    rician_K_linear = max_path_power / scatter_power;
    rician_K_dB = 10 * log10(rician_K_linear); % 莱斯 K 因子 (dB)
end
significant_paths_count = K; % 当前设定的显著多径数量

% --- (3) 频域特征 (Frequency Domain Parameters) ---
% 相干带宽估算: B_c \approx 1 / (5 * \sigma_\tau)
if rms_delay_spread > 0
    coherence_bandwidth = 1 / (5 * rms_delay_spread);
else
    coherence_bandwidth = Inf;
end

%% 5. 在命令行输出全维度参数化报告
fprintf('\n=================================================================\n');
fprintf('                 多径信道特征参数化提取报告                      \n');
fprintf('=================================================================\n');
fprintf('【整体拟合指标】\n');
fprintf('  - 恢复信号与原信号相关系数: %.4f\n', corr_val);
fprintf('  - 信道总增益 (Total Gain):  %.6f\n', total_gain);
fprintf('-----------------------------------------------------------------\n');
fprintf('【1. 时延扩展特征 (Time Dispersion)】\n');
fprintf('  - 平均附加时延 (Mean Delay, tau_bar): %.6f 秒 (%.2f ms)\n', mean_excess_delay, mean_excess_delay*1000);
fprintf('  - 均方根时延扩展 (RMS Spread, sigma): %.6f 秒 (%.2f ms)\n', rms_delay_spread, rms_delay_spread*1000);
fprintf('  - 最大附加时延 (Max Excess Delay):    %.6f 秒 (%.2f ms)\n', max_excess_delay, max_excess_delay*1000);
fprintf('-----------------------------------------------------------------\n');
fprintf('【2. 能量与衰落特征 (Energy & Fading)】\n');
fprintf('  - 莱斯 K 因子 (Rician K-Factor):      %.2f dB\n', rician_K_dB);
fprintf('  - 显著多径数量 (Significant Paths):   %d 条\n', significant_paths_count);
fprintf('-----------------------------------------------------------------\n');
fprintf('【3. 频域特征 (Frequency Domain)】\n');
fprintf('  - 相干带宽预估 (Coherence Bandwidth): %.2f Hz\n', coherence_bandwidth);
fprintf('-----------------------------------------------------------------\n');
fprintf('【4. 稀疏结构参数矩阵 [tau_i, A_i, theta_i]】\n');
fprintf('%-6s | %-14s | %-14s | %-14s\n', '径序号', '绝对延迟(秒)', '幅度 (A_i)', '相位 (rad)');
fprintf('-----------------------------------------------------------------\n');
for i = 1:K
    fprintf('%-6d | %-14.6f | %-14.6f | %-14.6f\n', i, tau_seconds(i), amplitudes(i), phases(i));
end
fprintf('=================================================================\n\n');

%% 6. 结果可视化
time_axis_y = (0:length(y)-1)/fs_0;

% 图 1：信道冲激响应与功率时延谱 (PDP)
figure('Name', '多径信道参数化特征展示', 'Position', [100, 100, 1000, 600]);
subplot(2, 1, 1);
stem(tau_seconds*1000, amplitudes, 'b', 'Marker', 'o', 'LineWidth', 1.2);
title(sprintf('信道冲激响应幅度与时延分布 (K = %d, RMS时延扩展=%.2f ms)', K, rms_delay_spread*1000));
xlabel('延迟时间 (ms)');
ylabel('多径幅度 (A_i)');
grid on;

subplot(2, 1, 2);
plot(time_axis_y*1000, y, 'k', 'LineWidth', 1, 'DisplayName', '原带多径信号');
hold on;
plot(time_axis_y*1000, y_est, 'r--', 'LineWidth', 1.2, 'DisplayName', 'OMP恢复信号');
title(sprintf('信号重构对比 (相关系数 = %.4f)', corr_val));
xlabel('时间 (ms)');
ylabel('幅值');
legend('Location', 'best');
grid on;