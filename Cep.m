%% 1. 读取音频文件
[filename, pathname] = uigetfile('*.wav', '选择WAV文件');
if isequal(filename, 0), return; end
[y, fs] = audioread(fullfile(pathname, filename));

% 如果是多声道，转为单声道
if size(y, 2) > 1, y = mean(y, 2); end

%% 2. 计算 Log-Mel 谱图
% 设置参数
winTime = 0.025;               % 窗长 25ms (典型值)
hopTime = 0.010;               % 帧移 10ms
numBands = 64;                 % Mel 滤波器组数量

% 计算 Mel 谱
% Window: 汉宁窗，OverlapLength: 重叠点数
[S, f_mel, t_mel] = melSpectrogram(y, fs, ...
    'Window', hann(round(winTime*fs), 'periodic'), ...
    'OverlapLength', round((winTime-hopTime)*fs), ...
    'NumBands', numBands);

% 转换为对数刻度 (dB)
S_log = 10 * log10(S + eps);

%% 3. 绘图
figure('Color', 'w', 'Position', [100, 100, 800, 500]);

% 绘制时域波形（参考）
subplot(2,1,1);
t = (0:length(y)-1)/fs;
plot(t, y);
title(['音频波形: ', filename], 'Interpreter', 'none');
xlabel('时间 (s)'); ylabel('幅度');
grid on;

% 绘制 Log-Mel 谱图
subplot(2,1,2);
imagesc(t_mel, f_mel, S_log); 
set(gca, 'YDir', 'normal'); % 修正纵轴方向，使低频在下
colormap('jet');
colorbar;
title('Log-Mel 谱图');
xlabel('时间 (s)'); ylabel('频率 (Hz)');