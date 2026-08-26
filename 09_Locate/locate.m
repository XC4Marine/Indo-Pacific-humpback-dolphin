%% 环境准备与路径定义
clc; clear; close all;

% 改变baseDir就可以
% 路径定义
baseDir = 'D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains\PulseTrain_545';
trainPath = fullfile(baseDir, 'PulseTrain.wav');
paramPath = fullfile(baseDir, 'PulseParameters.txt');

%% 1. 加载音频与参数
if ~exist(trainPath, 'file') || ~exist(paramPath, 'file')
    error('Chico, 找不到指定的文件，请检查路径。');
end

[trainData, fs] = audioread(trainPath);
trainData = double(trainData);
N = length(trainData);

% 读取 IPI 参数
df_param = readtable(paramPath, 'Delimiter', 'tab', 'VariableNamingRule', 'preserve');
ipi_ms = df_param.("IPI(ms)");
numPulses = height(df_param);

%% 2. 构造"理想脉冲序列"模板 (Synthetic Template)
fprintf('正在构造基于 IPI 的理想脉冲模板...\n');

% 计算每个脉冲相对于模板开头的样本位置
ipi_samples = round(ipi_ms(1:end-1) * fs / 1000); % ms 转样本数
pulse_rel_indices = [1; 1 + cumsum(ipi_samples)]; % 累加得到索引序列

% 创建模板向量 (长度刚好覆盖所有 IPI)
templateLen = pulse_rel_indices(end) + round(0.001 * fs); % 留 1ms 余量
synthetic_template = zeros(templateLen, 1);
synthetic_template(pulse_rel_indices) = 1; % 在脉冲位置置 1

%% 3. 计算实际信号的 TEO (作为互相关的底图)
fprintf('正在计算 TEO 能量...\n');
teo_val = zeros(size(trainData));
teo_val(2:end-1) = trainData(2:end-1).^2 - trainData(1:end-2) .* trainData(3:end);
% 进行归一化以提高互相关稳定性
teo_norm = teo_val / max(teo_val);

%% 4. 全局互相关对齐
fprintf('正在进行模板匹配 (全局互相关)...\n');
% 将"1序列模板"与信号的"TEO能量图"进行互相关
[c, lags] = xcorr(teo_norm, synthetic_template);

% 只允许全部 Pulse 模板点都落在音频范围内的 lag。
% 如果不加此约束，短 PulseTrain 可能让部分模板越界，并被局部强 TEO 峰吸引。
validLagMask = (pulse_rel_indices(1) + lags >= 1) & ...
    (pulse_rel_indices(end) + lags <= N);
if ~any(validLagMask)
    error('Chico, 不存在可让全部 Pulse 模板点落入音频范围的有效 lag。');
end

[unconstrainedBestScore, unconstrainedMaxIdx] = max(c);
validLagIndices = find(validLagMask);
[bestScore, validMaxIdx] = max(c(validLagMask));
maxIdx = validLagIndices(validMaxIdx);
bestLag = lags(maxIdx); % 完整模板约束下最契合的起始样本偏移

if ~validLagMask(unconstrainedMaxIdx)
    fprintf(['无约束最佳 lag=%d（得分 %.6f）会导致模板越界，', ...
        '已改用完整模板 lag=%d（得分 %.6f）。\n'], ...
        lags(unconstrainedMaxIdx), unconstrainedBestScore, bestLag, bestScore);
else
    fprintf('完整模板最佳 lag=%d（得分 %.6f）。\n', bestLag, bestScore);
end

% 根据偏移计算所有脉冲的最终理论时间点
theoretical_samples = pulse_rel_indices + bestLag;
theoretical_times = (theoretical_samples - 1) / fs;

%% 5. 自动检测流程 (FFT粗筛 + TEO精定位)
% (此处保留您之前要求的检测逻辑，用于在下图中对比)
fprintf('正在执行自动检测流程进行对比...\n');
fft_size = 512;
hop_size = 256;
win = hann(fft_size);
freqs = (0:fft_size/2) * (fs/fft_size);
idx_band = find(freqs >= 15000 & freqs <= 95000);
noise_baseline = prctile(teo_val, 40);
core_threshold = noise_baseline * 100;

detected_clicks = [];
reverseStr = '';
for i = 1 : hop_size : (N - fft_size)
    if mod(i, 10000) == 1
        msg = sprintf('检测进度: %3.1f%%', 100 * i / N);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    segment = trainData(i : i+fft_size-1) .* win;
    spec = abs(fft(segment));
    spec = spec(1:fft_size/2+1);
    db_spec = 20 * log10(spec ./ (mean(spec) + eps));
    
    if sum(db_spec(idx_band) > 13) > (length(idx_band) * 0.125)
        search_range = max(1, i-512) : min(N, i+1024);
        search_teo = teo_val(search_range);
        [val, peak_loc] = max(search_teo);
        if val > core_threshold
            peak_idx = search_range(1) + peak_loc - 1;
            % 简单的边界检测
            r_start = max(1, peak_idx-300); r_end = min(N, peak_idx+300);
            ma_teo = movmean(teo_val(r_start:r_end), 10);
            active = find(ma_teo > noise_baseline * 3);
            if ~isempty(active)
                new_click.start = r_start + active(1);
                new_click.end = r_start + active(end);
                new_click.peak = peak_idx;
                new_click.val = val;
                
                % 去重
                if isempty(detected_clicks) || (peak_idx - detected_clicks(end).peak > 0.01 * fs)
                    detected_clicks = [detected_clicks; new_click];
                end
            end
        end
    end
end
fprintf('\n处理完成。\n');

%% 6. 可视化对比
timeAxis = (0:N-1) / fs;
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);

% 子图1: 理想模板匹配结果 (波形图)
subplot(2,1,1);
plot(timeAxis, trainData, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5); hold on;
% 标注对齐后的脉冲位置
stem(theoretical_times, ones(size(theoretical_times))*max(trainData), ...
     'r', 'Marker', 'none', 'LineWidth', 1, 'DisplayName', 'Matched Template Pulse');
title('Subplot 1: Waveform with Global Matched Template (Based on IPI Spikes)');
ylabel('Amplitude');
legend('show');
grid on;

% 子图2: 自动检测结果 (TEO图)
subplot(2,1,2);
plot(timeAxis, teo_val, 'b'); hold on;
% 绘制检测到的脉冲色块
for j = 1:length(detected_clicks)
    px = [detected_clicks(j).start, detected_clicks(j).end, ...
          detected_clicks(j).end, detected_clicks(j).start] / fs;
    py = [0 0 max(teo_val) max(teo_val)];
    patch(px, py, 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
title(['Subplot 2: TEO Detection Results (Detected Count: ', num2str(length(detected_clicks)), ')']);
xlabel('Time (s)');
ylabel('TEO Energy');
grid on;

% 联动缩放
linkaxes(findobj(gcf, 'Type', 'axes'), 'x');
xlim([theoretical_times(1)-0.05, theoretical_times(end)+0.05]); % 默认聚焦到脉冲群区域

%% 7. 创建文件夹并提取前后 5ms 的波形与图片
% 以下旧版导出模块已停用：保留代码，但不再生成裁剪 WAV 或逐 Pulse PNG。
%{
% 定义保存路径
saveFolderName = 'Extracted_Pulses_5ms';
saveDirPath = fullfile(baseDir, saveFolderName);

% 创建文件夹
if ~exist(saveDirPath, 'dir')
    mkdir(saveDirPath);
    fprintf('已创建保存文件夹: %s\n', saveDirPath);
end

% 计算 5ms 对应的样本数
% 512kHz 下，5ms 为 2560 个采样点
halfWindowSamples = round(5e-3 * fs); 
t_segment_ms = ((-halfWindowSamples:halfWindowSamples) / fs) * 1000; % 转换为毫秒时间轴

fprintf('开始批量保存 .wav 和 .png 文件 (窗口: +/- 5ms)...\n');

% 进度条初始化
reverseStr = '';
numExtracted = length(theoretical_samples);

% 创建一个隐藏的 figure 句柄
hFig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 800 400]); 

for k = 1:numExtracted
    % 1. 计算截取索引
    centerIdx = round(theoretical_samples(k));
    startIdx = centerIdx - halfWindowSamples;
    endIdx = centerIdx + halfWindowSamples;
    
    % 边界检查：如果超出音频范围则进行填充或截断
    if startIdx < 1
        % 如果左侧超出，补零或只取有效部分
        pulseSegment = [zeros(abs(startIdx)+1, 1); trainData(1:endIdx)];
    elseif endIdx > N
        % 如果右侧超出
        pulseSegment = [trainData(startIdx:N); zeros(endIdx-N, 1)];
    else
        pulseSegment = trainData(startIdx:endIdx);
    end
    
    % 2. 保存 .wav 文件
    wavName = sprintf('Pulse_%03d.wav', k);
    audiowrite(fullfile(saveDirPath, wavName), pulseSegment, fs);
    
    % 3. 生成并保存 .png 图片
    clf(hFig);
    ax = axes(hFig);
    plot(ax, t_segment_ms, pulseSegment, 'LineWidth', 1, 'Color', [0.2 0.2 0.2]);
    hold(ax, 'on');
    
    % 在中心画一条红虚线标注脉冲位置
    xline(ax, 0, '--r', 'Alpha', 0.5);
    
    title(ax, sprintf('Pulse #%03d (Center: %.4f s, Window: \\pm 5ms)', k, theoretical_times(k)));
    xlabel(ax, 'Relative Time (ms)');
    ylabel(ax, 'Amplitude');
    grid(ax, 'on');
    
    % 锁定纵坐标范围
    ylim(ax, [-max(abs(trainData)) max(abs(trainData))]*1.1);
    
    imgName = sprintf('Pulse_%03d.png', k);
    saveas(hFig, fullfile(saveDirPath, imgName));
    
    % 4. 更新总体进度条
    percentDone = 100 * k / numExtracted;
    % 构建简单的字符进度条
    barLen = 20;
    filledLen = floor(barLen * k / numExtracted);
    barStr = [repmat('=', 1, filledLen), repmat(' ', 1, barLen - filledLen)];
    
    msg = sprintf('总体进度: [%s] %.1f%% (%d/%d)', barStr, percentDone, k, numExtracted);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end

close(hFig);
fprintf('\n处理完成！所有 5ms 片段已保存至: %s\n', saveDirPath);
%% 8. 模块8：保存每个 Pulse 的 Teager 算子与自相关分析图
% 定义保存路径
analysisFolderName = 'Pulse_Analysis_TEO_Autocorr';
analysisDirPath = fullfile(baseDir, analysisFolderName);

% 创建文件夹
if ~exist(analysisDirPath, 'dir')
    mkdir(analysisDirPath);
    fprintf('已创建分析文件夹: %s\n', analysisDirPath);
end

% 参数设置 (沿用 5ms 窗口)
halfWindowSamples = round(5e-3 * fs); 
t_segment_ms = ((-halfWindowSamples:halfWindowSamples) / fs) * 1000; % 毫秒时间轴

fprintf('开始生成 Teager 算子与自相关对比图...\n');

% 进度条初始化
reverseStr = '';
numExtracted = length(theoretical_samples);

% 创建一个隐藏的 figure
hFigAnalysis = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 600]);

for k = 1:numExtracted
    % 1. 提取数据 (逻辑同前，保证对齐)
    centerIdx = round(theoretical_samples(k));
    startIdx = centerIdx - halfWindowSamples;
    endIdx = centerIdx + halfWindowSamples;
    
    if startIdx < 1 || endIdx > N
        % 边界填充处理
        if startIdx < 1
            segment = [zeros(abs(startIdx)+1, 1); trainData(1:endIdx)];
        else
            segment = [trainData(startIdx:N); zeros(endIdx-N, 1)];
        end
    else
        segment = trainData(startIdx:endIdx);
    end
    
    % 2. 计算 Teager 能量算子 (TEO)
    % TEO 修正公式：Psi(x(n)) = x(n)^2 - x(n-1)x(n+1)
    teo_seg = zeros(size(segment));
    teo_seg(2:end-1) = segment(2:end-1).^2 - segment(1:end-2) .* segment(3:end);
    
    % 3. 计算自相关函数 (Autocorrelation)
    [acf, lags] = xcorr(segment, 'coeff');
    lags_ms = (lags / fs) * 1000; % 延迟转换成毫秒
    
    % 4. 绘图
    clf(hFigAnalysis);
    
    % 上子图：Teager 算子时域图
    subplot(2, 1, 1);
    plot(t_segment_ms, teo_seg, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1);
    grid on;
    title(sprintf('Pulse #%03d: Teager Energy Operator (TEO)', k));
    xlabel('Time (ms)');
    ylabel('TEO Energy');
    xlim([-5 5]);
    
    % 下子图：自相关函数图 (用于观察时延特征)
    subplot(2, 1, 2);
    plot(lags_ms, acf, 'Color', [0 0.4470 0.7410], 'LineWidth', 1);
    grid on;
    title(sprintf('Pulse #%03d: Autocorrelation (Lag Analysis)', k));
    xlabel('Delay (ms)');
    ylabel('Correlation Coeff');
    xlim([-5 5]); % 重点观察中心附近的延迟
    
    % 5. 保存图片
    analysisImgName = sprintf('Analysis_Pulse_%03d.png', k);
    saveas(hFigAnalysis, fullfile(analysisDirPath, analysisImgName));
    
    % 6. 更新总体进度条
    percentDone = 100 * k / numExtracted;
    barLen = 20;
    filledLen = floor(barLen * k / numExtracted);
    barStr = [repmat('=', 1, filledLen), repmat(' ', 1, barLen - filledLen)];
    
    msg = sprintf('模块8进度: [%s] %.1f%% (%d/%d)', barStr, percentDone, k, numExtracted);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end

close(hFigAnalysis);
fprintf('\n模块8处理完成！分析结果已保存至: %s\n', analysisDirPath);
%}

%% 9. Pulse 定位准确性 Report（不修改或重新裁剪 Pulse）
reportFolderName = 'Locate_Position_Report';
reportDirPath = fullfile(baseDir, reportFolderName);
if ~exist(reportDirPath, 'dir')
    mkdir(reportDirPath);
end

reportCsvPath = fullfile(reportDirPath, 'pulse_position_report.csv');
reportPngPath = fullfile(reportDirPath, 'pulse_position_summary.png');
reportTxtPath = fullfile(reportDirPath, 'pulse_position_summary.txt');

% 按参数表中的 Pulse 编号匹配原始 Pulse_N.wav，避免按文件名字典序错排。
pulseNumbers = df_param{:, "No.Pulse"};
pulseNumbers = double(pulseNumbers(:));
reportCount = numel(pulseNumbers);

algorithmSample = nan(reportCount, 1);
availableAlgorithmCount = min(reportCount, numel(theoretical_samples));
algorithmSample(1:availableAlgorithmCount) = ...
    double(theoretical_samples(1:availableAlgorithmCount));

referenceStartSample = nan(reportCount, 1);
referencePeakSample = nan(reportCount, 1);
matchStatus = strings(reportCount, 1);
referenceFile = strings(reportCount, 1);

fprintf('\n开始生成 Pulse 定位准确性 Report...\n');
reverseStr = '';
for k = 1:reportCount
    pulseNumber = pulseNumbers(k);
    pulseFileName = sprintf('Pulse_%d.wav', pulseNumber);
    pulsePath = fullfile(baseDir, pulseFileName);
    referenceFile(k) = string(pulseFileName);

    if ~exist(pulsePath, 'file')
        matchStatus(k) = "missing_reference_file";
    else
        [pulseData, pulseFs] = audioread(pulsePath);
        pulseData = double(pulseData);

        if pulseFs ~= fs
            matchStatus(k) = "sample_rate_mismatch";
        elseif size(pulseData, 2) ~= size(trainData, 2)
            matchStatus(k) = "channel_count_mismatch";
        elseif isempty(pulseData) || size(pulseData, 1) > N
            matchStatus(k) = "invalid_reference_length";
        else
            exactStarts = localFindExactWaveformStarts(trainData, pulseData);
            if isempty(exactStarts)
                matchStatus(k) = "no_exact_match";
            elseif numel(exactStarts) > 1
                matchStatus(k) = "multiple_exact_matches";
            else
                pulseTeo = zeros(size(pulseData, 1), 1);
                pulseMono = pulseData(:, 1);
                if numel(pulseMono) >= 3
                    pulseTeo(2:end-1) = pulseMono(2:end-1).^2 - ...
                        pulseMono(1:end-2) .* pulseMono(3:end);
                end
                [maxPulseTeo, localPeakIndex] = max(pulseTeo);
                if isempty(maxPulseTeo) || ~isfinite(maxPulseTeo) || maxPulseTeo <= 0
                    matchStatus(k) = "invalid_reference_teo";
                else
                    referenceStartSample(k) = exactStarts(1);
                    referencePeakSample(k) = exactStarts(1) + localPeakIndex - 1;
                    matchStatus(k) = "matched";
                end
            end
        end
    end

    percentDone = 100 * k / max(1, reportCount);
    barLen = 20;
    filledLen = floor(barLen * k / max(1, reportCount));
    barStr = [repmat('=', 1, filledLen), ...
        repmat(' ', 1, barLen - filledLen)];
    msg = sprintf('Report 进度: [%s] %.1f%% (%d/%d)', ...
        barStr, percentDone, k, reportCount);
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf('\n');

algorithmTimeS = (algorithmSample - 1) / fs;
referenceTimeS = (referencePeakSample - 1) / fs;
signedErrorSamples = algorithmSample - referencePeakSample;
absErrorSamples = abs(signedErrorSamples);
signedErrorMs = signedErrorSamples / fs * 1000;
absErrorMs = abs(signedErrorMs);

reportTable = table(pulseNumbers, algorithmSample, algorithmTimeS, ...
    referenceStartSample, referencePeakSample, referenceTimeS, ...
    signedErrorSamples, absErrorSamples, signedErrorMs, absErrorMs, ...
    matchStatus, referenceFile, ...
    'VariableNames', {'PulseNumber', 'AlgorithmSample', 'AlgorithmTime_s', ...
    'ReferenceStartSample', 'ReferencePeakSample', 'ReferenceTime_s', ...
    'SignedError_samples', 'AbsError_samples', 'SignedError_ms', ...
    'AbsError_ms', 'MatchStatus', 'ReferenceFile'});
writetable(reportTable, reportCsvPath, 'Encoding', 'UTF-8');

validMask = matchStatus == "matched" & isfinite(signedErrorMs);
validErrorsMs = signedErrorMs(validMask);
validAbsErrorsMs = absErrorMs(validMask);
validCount = nnz(validMask);

if validCount > 0
    meanBiasMs = mean(validErrorsMs);
    maeMs = mean(validAbsErrorsMs);
    medianAbsErrorMs = median(validAbsErrorsMs);
    rmseMs = sqrt(mean(validErrorsMs.^2));
    p95AbsErrorMs = prctile(validAbsErrorsMs, 95);
    maxAbsErrorMs = max(validAbsErrorsMs);
else
    meanBiasMs = NaN;
    maeMs = NaN;
    medianAbsErrorMs = NaN;
    rmseMs = NaN;
    p95AbsErrorMs = NaN;
    maxAbsErrorMs = NaN;
end

summaryFileId = fopen(reportTxtPath, 'w', 'n', 'UTF-8');
if summaryFileId < 0
    error('Chico, 无法创建 Report 摘要文件: %s', reportTxtPath);
end
summaryCleanup = onCleanup(@() fclose(summaryFileId));
fprintf(summaryFileId, 'Pulse 定位准确性 Report\n');
fprintf(summaryFileId, '==========================\n');
fprintf(summaryFileId, '输入目录: %s\n', baseDir);
fprintf(summaryFileId, '采样率: %.0f Hz\n', fs);
fprintf(summaryFileId, 'Pulse 总数: %d\n', reportCount);
fprintf(summaryFileId, '有效唯一匹配: %d\n', validCount);
fprintf(summaryFileId, '平均偏差: %.9f ms\n', meanBiasMs);
fprintf(summaryFileId, '平均绝对误差 MAE: %.9f ms\n', maeMs);
fprintf(summaryFileId, '中位绝对误差: %.9f ms\n', medianAbsErrorMs);
fprintf(summaryFileId, '均方根误差 RMSE: %.9f ms\n', rmseMs);
fprintf(summaryFileId, '绝对误差 95%% 分位数: %.9f ms\n', p95AbsErrorMs);
fprintf(summaryFileId, '最大绝对误差: %.9f ms\n', maxAbsErrorMs);
clear summaryCleanup;

reportFigure = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1200 900]);
reportLayout = tiledlayout(reportFigure, 3, 1, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(reportLayout, 1);
plot(timeAxis, teo_val, 'Color', [0.65 0.65 0.65], 'LineWidth', 0.5, ...
    'HandleVisibility', 'off');
hold on;
validReferenceTimes = referenceTimeS(validMask);
validAlgorithmTimes = algorithmTimeS(validMask);
referenceMarkerY = interp1(timeAxis, teo_val, validReferenceTimes, ...
    'nearest', NaN);
algorithmMarkerY = interp1(timeAxis, teo_val, validAlgorithmTimes, ...
    'nearest', NaN);
scatter(validReferenceTimes, referenceMarkerY, 28, [0.1 0.6 0.2], ...
    'filled', 'DisplayName', 'Reference TEO peak');
scatter(validAlgorithmTimes, algorithmMarkerY, 28, [0.85 0.15 0.15], ...
    'x', 'LineWidth', 1.2, 'DisplayName', 'IPI-template location');
xlabel('Time (s)');
ylabel('TEO Energy');
title('PulseTrain TEO: algorithm positions vs. exact-reference positions');
legend('Location', 'best');
grid on;

nexttile(reportLayout, 2);
plot(pulseNumbers(validMask), signedErrorMs(validMask), '-o', ...
    'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
yline(0, '--k');
xlabel('Pulse Number');
ylabel('Signed Error (ms)');
title('Per-pulse signed location error (algorithm - reference)');
grid on;

nexttile(reportLayout, 3);
if validCount > 0
    histogram(validAbsErrorsMs, 'FaceColor', [0.8500 0.3250 0.0980]);
else
    text(0.5, 0.5, 'No valid exact-reference matches', ...
        'HorizontalAlignment', 'center');
    xlim([0 1]);
    ylim([0 1]);
end
xlabel('Absolute Error (ms)');
ylabel('Pulse Count');
title(sprintf(['Absolute error distribution | Valid %d/%d | ', ...
    'MAE %.6f ms | RMSE %.6f ms'], ...
    validCount, reportCount, maeMs, rmseMs));
grid on;

title(reportLayout, 'Pulse Location Accuracy Report');
exportgraphics(reportFigure, reportPngPath, 'Resolution', 200);
close(reportFigure);

fprintf('Report 生成完成:\n');
fprintf('  CSV: %s\n', reportCsvPath);
fprintf('  PNG: %s\n', reportPngPath);
fprintf('  TXT: %s\n', reportTxtPath);
fprintf('  有效唯一匹配: %d/%d\n', validCount, reportCount);

%% Report 辅助函数
function exactStarts = localFindExactWaveformStarts(longSignal, shortSignal)
% 使用高幅值锚点缩小候选范围，再执行逐样本完全一致检查。
    longLength = size(longSignal, 1);
    shortLength = size(shortSignal, 1);
    exactStarts = zeros(0, 1);

    anchorSignal = shortSignal(:, 1);
    [~, anchorOrder] = sort(abs(anchorSignal), 'descend');
    anchorCount = min(6, shortLength);
    anchorOffsets = anchorOrder(1:anchorCount);

    firstOffset = anchorOffsets(1);
    candidateStarts = find(longSignal(:, 1) == anchorSignal(firstOffset)) ...
        - firstOffset + 1;
    candidateStarts = candidateStarts(candidateStarts >= 1 & ...
        candidateStarts + shortLength - 1 <= longLength);

    for anchorIndex = 2:numel(anchorOffsets)
        offset = anchorOffsets(anchorIndex);
        if isempty(candidateStarts)
            return;
        end
        keepMask = longSignal(candidateStarts + offset - 1, 1) == ...
            anchorSignal(offset);
        candidateStarts = candidateStarts(keepMask);
    end

    for candidateIndex = 1:numel(candidateStarts)
        candidateStart = candidateStarts(candidateIndex);
        candidateEnd = candidateStart + shortLength - 1;
        if isequal(longSignal(candidateStart:candidateEnd, :), shortSignal)
            exactStarts(end + 1, 1) = candidateStart; %#ok<AGROW>
        end
    end
end
