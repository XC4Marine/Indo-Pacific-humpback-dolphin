% 1. 读取音频信号
filename = 'FirstArrival_Train094_pulse4.wav'; % 替换为你的实际音频文件路径
[s, fs] = audioread(filename);

% 如果读取的是双声道音频，取其中一个声道即可
if size(s, 2) > 1
    s = s(:, 1);
end

% (可选) 如果你只想分析其中一小段 click 直达信号，可以在这里截取：
% s = s(50:100); 

% 2. 计算快速傅里叶变换
L = length(s);      % 获取当前信号的实际长度
Y = fft(s);         % FFT 变换

% 3. 提取相位信息
phase_rad = angle(Y);                 % 获取复数的相位角 (弧度)
P1_phase = phase_rad(1:floor(L/2)+1); % 截取单侧频段 (0 到 奈奎斯特频率)

% 构造对应的频率轴
f = fs * (0:floor(L/2)) / L/1000;

% 4. 绘制连续相位谱
figure('Name', '信号相位谱', 'Position', [100, 100, 800, 400]);
plot(f, unwrap(P1_phase), 'b', 'LineWidth', 1.2); 
title('单侧连续相位谱 (Unwrapped Phase)');
xlabel('频率 (kHz)');
ylabel('连续相位 (弧度)');
grid on;

%%

% 2. 计算群延迟
% 将 s 视为 FIR 滤波器的分子系数，分母系数为 1
% gd 返回的群延迟单位为"采样点数" (Samples)，f 为对应的物理频率 (Hz)
[gd, f] = grpdelay(s, 1, N, fs);

% 将群延迟从"采样点数"转换为"秒" (s)
gd_sec = gd / fs;

% 由于高频环境噪声会导致相位发生剧烈卷绕，计算特征前通常需要提取有效频带
% 假设目标信号(如宽带 Click 脉冲)的主要能量集中在 10kHz 到 150kHz
valid_idx = (f >= 20000 & f <= 230000);
f_valid = f(valid_idx);
gd_valid = gd_sec(valid_idx);

% 可选：对群延迟进行平滑处理以减少随机相位噪声造成的极端波动
gd_smoothed = smoothdata(gd_valid, 'sgolay', 15);

% 3. 提取特征参数
% 特征 A：平均群延迟 (秒)
mean_gd = mean(gd_smoothed);

% 特征 B：群延迟波动方差
var_gd = var(gd_smoothed);

% 特征 C：最大群延迟所在频率 (Hz)
[~, max_idx] = max(gd_smoothed);
max_gd_freq = f_valid(max_idx);

% 4. 可视化检查
figure;
plot(f_valid / 1000, gd_smoothed * 1e6, 'b', 'LineWidth', 1.5);
hold on;
plot(max_gd_freq / 1000, gd_smoothed(max_idx) * 1e6, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Frequency (kHz)');
ylabel('Group Delay (\mu s)');
title('Smoothed Group Delay of the Audio Signal');
grid on;