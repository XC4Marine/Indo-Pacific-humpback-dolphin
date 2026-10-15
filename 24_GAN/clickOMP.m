% 使用 OMP (Orthogonal Matching Pursuit) 提取多径信道冲激响应 (K=3~100 趋势与关键点分析)
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

%% 3. OMP 算法核心实现 (迭代到最大 K=100)
K_max = 100; 
residual = y; 
support = []; 

% 预分配空间，用于存储 K=3~100 的结果
h_est_all = zeros(N, K_max); 
y_est_all = zeros(length(y), K_max);
corr_all = zeros(1, K_max); % 存储相关系数

fprintf('开始 OMP 迭代计算 (最大迭代次数 %d)...\n', K_max);

for iter = 1:K_max
    % 1. 计算残差与字典原子的内积
    correlations = Phi_normalized' * residual;
    
    % 2. 找到最匹配的路径
    [~, best_idx] = max(abs(correlations));
    support = [support, best_idx];
    
    % 3. 最小二乘更新信道系数
    Phi_sub = Phi(:, support);
    h_sub = Phi_sub \ y;
    
    % 4. 更新残差
    residual = y - Phi_sub * h_sub;
    
    % 5. 记录当前 K 的信道估计结果
    h_est_temp = zeros(N, 1);
    h_est_temp(support) = h_sub;
    h_est_all(:, iter) = h_est_temp;
    
    % 6. 计算恢复的信号
    y_est_temp = Phi * h_est_temp; 
    y_est_all(:, iter) = y_est_temp;
    
    % 7. 计算与原信号的相关系数 (Pearson correlation)
    R = corrcoef(y, y_est_temp);
    corr_all(iter) = R(1, 2);
    
    if mod(iter, 10) == 0
        fprintf('已完成 K = %d 迭代，当前相关性: %.4f\n', iter, corr_all(iter));
    end
end

fprintf('迭代完成。正在生成可视化图表...\n');

%% 4. 结果可视化

% --- 图 1：相关性随多径数量 K (3~100) 变化的整体趋势曲线 ---
figure('Name', '相关性趋势分析 (K=3~100)', 'Position', [100, 100, 800, 450]);
plot(3:100, corr_all(3:100), '-bo', 'LineWidth', 1.5, 'MarkerSize', 3, 'MarkerFaceColor', 'b');
title('恢复信号与原信号的相关性随多径数 (K=3~100) 变化趋势');
xlabel('预测多径数量 (K)');
ylabel('相关系数 (Correlation)');
grid on;

% --- 图 2：选择关键里程碑节点 (K = 10, 30, 60, 100) 对比信道与恢复波形 ---
key_ks = [10, 30, 60, 100];
time_axis_y = (0:length(y)-1)/fs_0;

figure('Name', '关键 K 值下的信道与恢复波形对比', 'Position', [150, 150, 1200, 800]);
for i = 1:length(key_ks)
    k = key_ks(i);
    
    % 左侧：信道冲激响应
    subplot(4, 2, (i-1)*2 + 1);
    stem((0:N-1)/fs, h_est_all(:, k), 'b', 'Marker', 'none', 'LineWidth', 1);
    title(sprintf('K = %d 信道冲激响应 (相关性=%.4f)', k, corr_all(k)), 'FontSize', 10);
    ylabel('幅度');
    grid on;
    
    % 右侧：信号恢复对比
    subplot(4, 2, (i-1)*2 + 2);
    plot(time_axis_y, y, 'k', 'LineWidth', 0.8, 'DisplayName', '原带多径信号');
    hold on;
    plot(time_axis_y, y_est_all(:, k), 'r--', 'LineWidth', 1, 'DisplayName', '恢复信号');
    title(sprintf('K = %d 恢复信号对比', k), 'FontSize', 10);
    legend('Location', 'best');
    grid on;
end
sgtitle('OMP 关键多径数量 (K) 状态下的信道与信号重构对比');