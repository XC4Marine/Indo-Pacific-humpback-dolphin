% 1. 读取音频文件
% 请确保两个音频的采样率一致，这里假设它们具有相同的采样率 fs
file1 = 'Train094_Pulse_004.wav'; % 替换为你的第一个音频文件路径
file2 = 'FirstArrival_Train094_pulse4.wav'; % 替换为你的第二个音频文件路径

[x1, fs] = audioread(file1);
[x2, fs2] = audioread(file2);

% 如果音频采样率不同，需要进行重采样 (这里做一个简单的防错提示)
if fs ~= fs2
    error('两个音频的采样率不一致，请先使用 resample 函数统一采样率。');
end

% 2. 转换为单声道 (如果是双声道音频，取其中一个声道即可)
if size(x1, 2) > 1
    x1 = x1(:, 1);
end
if size(x2, 2) > 1
    x2 = x2(:, 1);
end

% 为了计算速度，如果音频太长，可以只截取前面一部分，例如前10秒
% max_len = 10 * fs;
% x1 = x1(1:min(length(x1), max_len));
% x2 = x2(1:min(length(x2), max_len));

% 3. 计算自相关和互相关
% xcorr 计算互相关，返回值 R 包含相关系数，lags 包含对应的延迟样本数
[R_xx, lags_xx] = xcorr(x1);       % 音频1的自相关
[R_xy, lags_xy] = xcorr(x1, x2);   % 音频1和音频2的互相关

% 将延迟的样本数转换为时间 (秒)
time_xx = lags_xx / fs;
time_xy = lags_xy / fs;

% 4. 绘制图像
figure('Name', '音频相关性分析', 'Position', [100, 100, 800, 600]);

% 子图 1: 自相关
subplot(2, 1, 1);
plot(time_xx, R_xx, 'b');
title('音频 1 的自相关 (Autocorrelation)');
xlabel('延迟时间 (秒)');
ylabel('相关幅值');
grid on;

% 子图 2: 互相关
subplot(2, 1, 2);
plot(time_xy, R_xy, 'r');
title('音频 1 与音频 2 的互相关 (Cross-correlation)');
xlabel('延迟时间 (秒)');
ylabel('相关幅值');
grid on;

% 调整布局使其更美观
sgtitle('音频自相关与互相关分析结果');


