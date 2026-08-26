function app = ManualMultipathAnnotator(varargin)
%MANUALMULTIPATHANNOTATOR Manually annotate direct and multipath arrivals.
%
%   ManualMultipathAnnotator() opens the annotation GUI with the project
%   defaults used by the 2026-08-05 recall-oriented multipath workflow.
%
%   ManualMultipathAnnotator('DiagnosticsRoot', PATH, 'AudioRoot', PATH,
%   'OutputCsv', PATH) overrides the input and output locations.
%
%   REPORT = ManualMultipathAnnotator('RunSelfTest', true) performs
%   non-interactive numerical, mapping, and CSV-schema checks.
%
%   CSV sample positions are zero-based so that they agree with the
%   existing Python selected_paths.csv files. MATLAB array access therefore
%   uses sample + 1 internally.

    projectRoot = 'D:\Project_Github\Indo-Pacific-humpback-dolphin';
    defaultDiagnostics = fullfile(projectRoot, '17_20260805', ...
        'multipath_results_3500_recall', 'diagnostics');
    defaultAudio = fullfile(projectRoot, '00_Data', 'Positive_Clicks_3500');
    defaultOutput = fullfile(projectRoot, '17_20260805', ...
        'multipath_results_3500_recall', 'manual_multipath_annotations.csv');

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'DiagnosticsRoot', defaultDiagnostics, @localIsTextScalar);
    addParameter(parser, 'AudioRoot', defaultAudio, @localIsTextScalar);
    addParameter(parser, 'OutputCsv', defaultOutput, @localIsTextScalar);
    addParameter(parser, 'RunSelfTest', false, @(x) islogical(x) && isscalar(x));
    parse(parser, varargin{:});

    diagnosticsRoot = char(string(parser.Results.DiagnosticsRoot));
    audioRoot = char(string(parser.Results.AudioRoot));
    outputCsv = char(string(parser.Results.OutputCsv));

    if parser.Results.RunSelfTest
        report = localRunSelfTests(diagnosticsRoot, audioRoot);
        if nargout > 0
            app = report;
        else
            disp(report);
        end
        return;
    end

    fileInfo = localBuildDataset(diagnosticsRoot, audioRoot, true);
    annotations = localCreateAnnotationTable(fileInfo);
    annotations = localLoadExistingAnnotations(annotations, outputCsv);
    savedAnnotations = annotations;
    autoLabels = localLoadAutomaticLabels(diagnosticsRoot);

    currentIndex = localFirstIncompleteIndex(annotations);
    currentTarget = "main";
    currentSignal = struct();
    isDirty = false;
    undoStack = {};
    signalCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    isClosing = false;

    colors.main = [0.0000, 0.4470, 0.7410];
    colors.multipath1 = [0.8500, 0.3250, 0.0980];
    colors.multipath2 = [0.4940, 0.1840, 0.5560];
    colors.automatic = [0.45, 0.45, 0.45];

    fig = uifigure('Name', '主径/多径人工标注器', ...
        'Position', [40, 30, 1720, 980], ...
        'Color', [0.97, 0.97, 0.97], ...
        'KeyPressFcn', @onKeyPress, ...
        'CloseRequestFcn', @onCloseRequested);

    rootGrid = uigridlayout(fig, [2, 2]);
    rootGrid.RowHeight = {74, '1x'};
    rootGrid.ColumnWidth = {'1x', 380};
    rootGrid.Padding = [12, 10, 12, 12];
    rootGrid.RowSpacing = 8;
    rootGrid.ColumnSpacing = 10;

    headerGrid = uigridlayout(rootGrid, [2, 4]);
    headerGrid.Layout.Row = 1;
    headerGrid.Layout.Column = [1, 2];
    headerGrid.RowHeight = {28, 28};
    headerGrid.ColumnWidth = {'2x', '1x', 260, 110};
    headerGrid.Padding = [8, 4, 8, 4];
    headerGrid.BackgroundColor = [1, 1, 1];

    fileLabel = uilabel(headerGrid, 'Text', '', 'FontWeight', 'bold', ...
        'FontSize', 13, 'Interpreter', 'none');
    fileLabel.Layout.Row = 1;
    fileLabel.Layout.Column = [1, 2];

    sampleLabel = uilabel(headerGrid, 'Text', '', 'HorizontalAlignment', 'right');
    sampleLabel.Layout.Row = 1;
    sampleLabel.Layout.Column = 3;

    dirtyLabel = uilabel(headerGrid, 'Text', '已保存', ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    dirtyLabel.Layout.Row = 1;
    dirtyLabel.Layout.Column = 4;

    messageLabel = uilabel(headerGrid, 'Text', '就绪', ...
        'FontColor', [0.20, 0.20, 0.20], 'Interpreter', 'none');
    messageLabel.Layout.Row = 2;
    messageLabel.Layout.Column = [1, 2];

    progressGauge = uigauge(headerGrid, 'linear', 'Limits', [0, 100], ...
        'MajorTicks', [0, 25, 50, 75, 100], 'MinorTicks', []);
    progressGauge.Layout.Row = 2;
    progressGauge.Layout.Column = 3;

    progressLabel = uilabel(headerGrid, 'Text', '', ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    progressLabel.Layout.Row = 2;
    progressLabel.Layout.Column = 4;

    plotGrid = uigridlayout(rootGrid, [3, 2]);
    plotGrid.Layout.Row = 2;
    plotGrid.Layout.Column = 1;
    plotGrid.RowHeight = {'1x', '1x', '1x'};
    plotGrid.ColumnWidth = {'1x', '1x'};
    plotGrid.Padding = [0, 0, 0, 0];
    plotGrid.RowSpacing = 6;
    plotGrid.ColumnSpacing = 8;

    waveformAxes = uiaxes(plotGrid);
    waveformAxes.Layout.Row = 1;
    waveformAxes.Layout.Column = 1;
    title(waveformAxes, '滤波波形');
    xlabel(waveformAxes, '零基采样点');
    ylabel(waveformAxes, '归一化幅值');
    grid(waveformAxes, 'on');
    waveformAxes.ButtonDownFcn = @(~, ~) onPlotClicked('sample', waveformAxes);

    rmsAxes = uiaxes(plotGrid);
    rmsAxes.Layout.Row = 1;
    rmsAxes.Layout.Column = 2;
    title(rmsAxes, '局部 RMS 能量包络（20 μs 滑窗，归一化）');
    xlabel(rmsAxes, '零基采样点');
    ylabel(rmsAxes, '归一化 RMS');
    grid(rmsAxes, 'on');
    rmsAxes.ButtonDownFcn = @(~, ~) onPlotClicked('sample', rmsAxes);

    phaseAxes = uiaxes(plotGrid);
    phaseAxes.Layout.Row = 3;
    phaseAxes.Layout.Column = 1;
    title(phaseAxes, '延迟补偿相对相位谱（选择主径和多径后显示）');
    xlabel(phaseAxes, '频率（kHz）');
    ylabel(phaseAxes, '多径相对直达径相位（°）');
    grid(phaseAxes, 'on');

    acfAxes = uiaxes(plotGrid);
    acfAxes.Layout.Row = 2;
    acfAxes.Layout.Column = 1;
    title(acfAxes, '归一化相关（0 居中，正时延可点选）');
    xlabel(acfAxes, '时延（采样点）');
    ylabel(acfAxes, '相关系数');
    grid(acfAxes, 'on');
    acfAxes.ButtonDownFcn = @(~, ~) onPlotClicked('lag', acfAxes);

    cepstrumAxes = uiaxes(plotGrid);
    cepstrumAxes.Layout.Row = 2;
    cepstrumAxes.Layout.Column = 2;
    title(cepstrumAxes, '完整实倒谱（正倒频率可点选多径）');
    xlabel(cepstrumAxes, '倒频率（ms）');
    ylabel(cepstrumAxes, '实倒谱幅值');
    grid(cepstrumAxes, 'on');
    cepstrumAxes.ButtonDownFcn = @(~, ~) onPlotClicked('quefrency', cepstrumAxes);

    spectrumAxes = uiaxes(plotGrid);
    spectrumAxes.Layout.Row = 3;
    spectrumAxes.Layout.Column = 2;
    title(spectrumAxes, '频谱包络（选择主径后显示，中心前后各 50 μs）');
    xlabel(spectrumAxes, '频率（kHz）');
    ylabel(spectrumAxes, '相对幅度（dB；主径峰值 = 0 dB）');
    grid(spectrumAxes, 'on');
    linkaxes([waveformAxes, rmsAxes], 'x');

    sidePanel = uipanel(rootGrid, 'Title', '标注控制', 'FontWeight', 'bold', ...
        'Scrollable', 'on', 'BackgroundColor', [1, 1, 1]);
    sidePanel.Layout.Row = 2;
    sidePanel.Layout.Column = 2;

    sideGrid = uigridlayout(sidePanel, [17, 1]);
    sideGrid.RowHeight = {84, 22, 34, 22, 32, 64, 22, 92, ...
        34, 34, 34, 34, 34, 34, 26, 26, '1x'};
    sideGrid.Padding = [10, 8, 10, 10];
    sideGrid.RowSpacing = 7;

    targetGroup = uibuttongroup(sideGrid, 'Title', '当前点选目标', ...
        'SelectionChangedFcn', @onTargetChanged);
    mainRadio = uiradiobutton(targetGroup, 'Text', '1 主径', 'Tag', 'main', ...
        'Position', [10, 10, 85, 24]);
    mp1Radio = uiradiobutton(targetGroup, 'Text', '2 多径1', 'Tag', 'multipath1', ...
        'Position', [105, 10, 90, 24]);
    mp2Radio = uiradiobutton(targetGroup, 'Text', '3 多径2', 'Tag', 'multipath2', ...
        'Position', [205, 10, 90, 24]);
    targetGroup.SelectedObject = mainRadio;

    uilabel(sideGrid, 'Text', '选择要查看的样本');
    sampleSelector = uidropdown(sideGrid, ...
        'Items', cellstr(localSampleSelectorItems(fileInfo)), ...
        'ItemsData', num2cell((1:height(fileInfo))'), ...
        'Value', currentIndex, 'ValueChangedFcn', @onSampleSelected);

    uilabel(sideGrid, 'Text', '样本状态');
    statusDropdown = uidropdown(sideGrid, ...
        'Items', {'未标注', '已完成（含多径）', '无可用多径', '不确定', '跳过'}, ...
        'ItemsData', {'unlabeled', 'completed', 'no_multipath', 'uncertain', 'skipped'}, ...
        'Value', 'unlabeled', 'ValueChangedFcn', @onStatusChanged);

    qualityGrid = uigridlayout(sideGrid, [2, 2]);
    qualityGrid.RowHeight = {22, 32};
    qualityGrid.ColumnWidth = {'1x', '1x'};
    qualityGrid.Padding = [0, 0, 0, 0];
    qualityGrid.ColumnSpacing = 8;
    mp1QualityLabel = uilabel(qualityGrid, 'Text', '多径1质量');
    mp1QualityLabel.Layout.Row = 1;
    mp1QualityLabel.Layout.Column = 1;
    mp2QualityLabel = uilabel(qualityGrid, 'Text', '多径2质量');
    mp2QualityLabel.Layout.Row = 1;
    mp2QualityLabel.Layout.Column = 2;
    qualityItems = {'不适用', '未补充', '清晰多径', '混叠多径', ...
        '混响尾部', '不确定', '1.5 ms后排除'};
    qualityItemsData = {'not_applicable', 'unlabeled', 'clean', 'overlapping', ...
        'reverberant', 'uncertain', 'exclude_late'};
    mp1QualityDropdown = uidropdown(qualityGrid, 'Items', qualityItems, ...
        'ItemsData', qualityItemsData, 'Value', 'not_applicable', ...
        'ValueChangedFcn', @(~, ~) onQualityChanged(1));
    mp1QualityDropdown.Layout.Row = 2;
    mp1QualityDropdown.Layout.Column = 1;
    mp2QualityDropdown = uidropdown(qualityGrid, 'Items', qualityItems, ...
        'ItemsData', qualityItemsData, 'Value', 'not_applicable', ...
        'ValueChangedFcn', @(~, ~) onQualityChanged(2));
    mp2QualityDropdown.Layout.Row = 2;
    mp2QualityDropdown.Layout.Column = 2;

    uilabel(sideGrid, 'Text', '备注');
    notesArea = uitextarea(sideGrid, 'Placeholder', '可填写判断依据或异常说明', ...
        'ValueChangedFcn', @onNotesChanged);

    uibutton(sideGrid, 'push', 'Text', '清除当前标签（Backspace）', ...
        'ButtonPushedFcn', @onClearCurrent);
    undoButton = uibutton(sideGrid, 'push', 'Text', '撤销上一步（Ctrl+Z）', ...
        'ButtonPushedFcn', @onUndo);

    navigationGrid = uigridlayout(sideGrid, [1, 2]);
    navigationGrid.ColumnWidth = {'1x', '1x'};
    navigationGrid.Padding = [0, 0, 0, 0];
    previousButton = uibutton(navigationGrid, 'push', 'Text', '← 上一条', ...
        'ButtonPushedFcn', @(~, ~) navigateBy(-1));
    nextButton = uibutton(navigationGrid, 'push', 'Text', '下一条 →', ...
        'ButtonPushedFcn', @(~, ~) navigateBy(1));

    uibutton(sideGrid, 'push', 'Text', '保存（Ctrl+S）', ...
        'FontWeight', 'bold', 'ButtonPushedFcn', @(~, ~) saveAnnotations(false));
    uibutton(sideGrid, 'push', 'Text', '保存并下一条', ...
        'FontWeight', 'bold', 'BackgroundColor', [0.18, 0.55, 0.34], ...
        'FontColor', [1, 1, 1], 'ButtonPushedFcn', @onSaveAndNext);

    fileGrid = uigridlayout(sideGrid, [1, 2]);
    fileGrid.ColumnWidth = {'1x', '1x'};
    fileGrid.Padding = [0, 0, 0, 0];
    uibutton(fileGrid, 'push', 'Text', '选择 diagnostics', ...
        'ButtonPushedFcn', @onChooseDiagnostics);
    uibutton(fileGrid, 'push', 'Text', '导出 CSV', ...
        'ButtonPushedFcn', @onExportCsv);

    optionsGrid = uigridlayout(sideGrid, [1, 2]);
    optionsGrid.ColumnWidth = {'1x', '1x'};
    optionsGrid.Padding = [0, 0, 0, 0];
    automaticCheckbox = uicheckbox(optionsGrid, 'Text', '显示自动参考', ...
        'Value', false, 'ValueChangedFcn', @(~, ~) redrawPlots());
    fullAcfCheckbox = uicheckbox(optionsGrid, 'Text', '完整相关范围', ...
        'Value', false, 'ValueChangedFcn', @(~, ~) redrawPlots());

    annotationSummary = uilabel(sideGrid, 'Text', '', 'FontWeight', 'bold', ...
        'WordWrap', 'on');
    uilabel(sideGrid, 'Text', ...
        ['波形/局部 RMS：点绝对时间位置。相关图和实倒谱：0 位于中央，' ...
         '先选主径，再点正时延或正倒频率。' ...
         '频谱图以主径峰值为共同 0 dB，保留各径相对幅度；同色虚线标出各自主峰的 −3 dB 带宽。' ...
         '相位图按标注中心对齐直达径与多径的前后各 50 μs 片段，显示多径相对直达径的互谱相位；' ...
         '仅保留两径幅度均超过各自峰值 10% 且高于 20 kHz 的频点。接近 ±180° 表示反相，' ...
         '可作为海面反射候选证据，但不能单独排除具有复杂反射相位的海底路径。' ...
         '多径图例同时给出其与主径线性幅度谱的 Pearson 相关系数。' ...
         '零与负时延仅供观察；多径不会自动吸附峰值。质量标签：清晰多径用于主分析，' ...
         '混叠多径用于路径重叠情形，混响尾部单独保留，不确定不进入正负样本，' ...
         '超过1.5 ms标记排除。'], ...
        'WordWrap', 'on', 'VerticalAlignment', 'top', ...
        'FontColor', [0.28, 0.28, 0.28]);

    loadCurrentSample();

    if nargout > 0
        app = struct('Figure', fig, 'SampleSelector', sampleSelector, ...
            'PhaseAxes', phaseAxes, 'CepstrumAxes', cepstrumAxes, ...
            'DiagnosticsRoot', diagnosticsRoot, 'AudioRoot', audioRoot, ...
            'OutputCsv', outputCsv);
    end

    function loadCurrentSample()
        if isempty(fileInfo) || currentIndex < 1 || currentIndex > height(fileInfo)
            return;
        end

        sourcePath = char(fileInfo.source_path(currentIndex));
        messageLabel.Text = sprintf('正在加载：%s', sourcePath);
        drawnow limitrate;
        try
            if isKey(signalCache, sourcePath)
                currentSignal = signalCache(sourcePath);
            else
                currentSignal = localPrepareSignal(sourcePath);
                signalCache(sourcePath) = currentSignal;
            end
        catch exception
            uialert(fig, getReport(exception, 'extended', 'hyperlinks', 'off'), ...
                '音频处理失败', 'Icon', 'error');
            messageLabel.Text = '音频处理失败';
            return;
        end

        undoStack = {};
        currentTarget = "main";
        targetGroup.SelectedObject = mainRadio;
        statusDropdown.Value = char(annotations.status(currentIndex));
        mp1QualityDropdown.Value = char(annotations.multipath1_quality(currentIndex));
        mp2QualityDropdown.Value = char(annotations.multipath2_quality(currentIndex));
        noteValue = annotations.notes(currentIndex);
        if ismissing(noteValue)
            noteValue = "";
        end
        notesArea.Value = cellstr(noteValue);
        isDirty = ~localRowsEqual(annotations, savedAnnotations);
        refreshUi();
        messageLabel.Text = '就绪：请选择标签目标后在图中点选';
    end

    function refreshUi()
        row = fileInfo(currentIndex, :);
        fileLabel.Text = sprintf('%s / %s / %s', row.voyage, row.pulse_train, row.click);
        sampleLabel.Text = sprintf('%d / %d   Fs=%d Hz   N=%d', ...
            currentIndex, height(fileInfo), currentSignal.fs, currentSignal.sampleCount);

        completedCount = sum(localAnnotationCompleteMask(annotations));
        percentage = 100 * completedCount / max(1, height(annotations));
        progressGauge.Value = percentage;
        progressLabel.Text = sprintf('%d/%d', completedCount, height(annotations));
        if isDirty
            dirtyLabel.Text = '未保存';
            dirtyLabel.FontColor = [0.80, 0.20, 0.10];
        else
            dirtyLabel.Text = '已保存';
            dirtyLabel.FontColor = [0.10, 0.50, 0.20];
        end
        previousButton.Enable = localOnOff(currentIndex > 1);
        nextButton.Enable = localOnOff(currentIndex < height(fileInfo));
        sampleSelector.Value = currentIndex;
        sampleSelector.Tooltip = char(fileInfo.relative_path(currentIndex));
        undoButton.Enable = localOnOff(~isempty(undoStack));
        mp1QualityDropdown.Enable = localOnOff(isfinite(annotations.multipath1_sample(currentIndex)));
        mp2QualityDropdown.Enable = localOnOff(isfinite(annotations.multipath2_sample(currentIndex)));
        annotationSummary.Text = localAnnotationSummary(annotations(currentIndex, :), currentSignal.fs);
        redrawPlots();
    end

    function redrawPlots()
        if isempty(fieldnames(currentSignal))
            return;
        end

        xSamples = 0:(currentSignal.sampleCount - 1);
        cla(waveformAxes);
        cla(rmsAxes);
        cla(acfAxes);
        cla(phaseAxes);
        cla(cepstrumAxes);
        delete(findall(spectrumAxes, 'Type', 'ConstantLine'));
        cla(spectrumAxes);
        legend(spectrumAxes, 'off');

        waveLine = plot(waveformAxes, xSamples, currentSignal.filtered, ...
            'Color', [0.16, 0.18, 0.20], 'LineWidth', 0.75);
        waveLine.ButtonDownFcn = @(~, ~) onPlotClicked('sample', waveformAxes);
        waveLine.PickableParts = 'all';
        title(waveformAxes, sprintf('滤波波形 — %s', fileInfo.relative_path(currentIndex)), ...
            'Interpreter', 'none');
        xlabel(waveformAxes, '零基采样点');
        ylabel(waveformAxes, '归一化幅值');
        grid(waveformAxes, 'on');
        xlim(waveformAxes, [0, max(1, currentSignal.sampleCount - 1)]);

        rmsLine = plot(rmsAxes, xSamples, currentSignal.rmsEnvelope, ...
            'Color', [0.30, 0.56, 0.23], 'LineWidth', 0.8);
        rmsLine.ButtonDownFcn = @(~, ~) onPlotClicked('sample', rmsAxes);
        rmsLine.PickableParts = 'all';
        title(rmsAxes, sprintf('局部 RMS 能量包络（%d 点，20 μs，归一化）', ...
            currentSignal.rmsWindowSamples));
        xlabel(rmsAxes, '零基采样点');
        ylabel(rmsAxes, '归一化 RMS');
        grid(rmsAxes, 'on');
        xlim(rmsAxes, [0, max(1, currentSignal.sampleCount - 1)]);

        lagSamples = currentSignal.acfLags;
        acfLine = plot(acfAxes, lagSamples, currentSignal.acf, ...
            'Color', [0.20, 0.45, 0.70], 'LineWidth', 0.85);
        acfLine.ButtonDownFcn = @(~, ~) onPlotClicked('lag', acfAxes);
        acfLine.PickableParts = 'all';
        xlabel(acfAxes, sprintf('时延（采样点；1 ms = %.0f 点）', currentSignal.fs / 1000));
        ylabel(acfAxes, '相关系数');
        grid(acfAxes, 'on');
        if fullAcfCheckbox.Value
            xlim(acfAxes, [lagSamples(1), lagSamples(end)]);
        else
            maxLag = min(currentSignal.sampleCount - 1, round(0.002 * currentSignal.fs));
            xlim(acfAxes, [-max(1, maxLag), max(1, maxLag)]);
        end
        xline(acfAxes, 0, '-', '0', 'Color', [0.25, 0.25, 0.25], ...
            'LineWidth', 1.0, 'HitTest', 'off');

        cepstrumLine = plot(cepstrumAxes, currentSignal.quefrencyMs, ...
            currentSignal.realCepstrum, 'Color', [0.10, 0.45, 0.70], ...
            'LineWidth', 0.85);
        cepstrumLine.ButtonDownFcn = @(~, ~) onPlotClicked('quefrency', cepstrumAxes);
        cepstrumLine.PickableParts = 'all';
        title(cepstrumAxes, '完整实倒谱（正倒频率可点选多径）');
        xlabel(cepstrumAxes, '倒频率（ms）');
        ylabel(cepstrumAxes, '实倒谱幅值');
        grid(cepstrumAxes, 'on');
        maxQuefrencyMs = min(2, currentSignal.quefrencyMs(end));
        xlim(cepstrumAxes, [-maxQuefrencyMs, maxQuefrencyMs]);
        xline(cepstrumAxes, 0, '-', '0', 'Color', [0.25, 0.25, 0.25], ...
            'LineWidth', 1.0, 'HitTest', 'off');

        row = annotations(currentIndex, :);
        drawManualMarkers(row);
        drawSpectralEnvelopes(row);
        drawRelativePhaseSpectrum(row);
        if automaticCheckbox.Value
            drawAutomaticMarkers();
        end
    end

    function drawManualMarkers(row)
        markerNames = {'main_sample', 'multipath1_sample', 'multipath2_sample'};
        markerLabels = {'主径', '多径1', '多径2'};
        markerColors = {colors.main, colors.multipath1, colors.multipath2};

        for markerIndex = 1:3
            sample = row.(markerNames{markerIndex});
            if ~isfinite(sample)
                continue;
            end
            color = markerColors{markerIndex};
            label = markerLabels{markerIndex};
            xline(waveformAxes, sample, '-', label, 'Color', color, ...
                'LineWidth', 1.7, 'LabelVerticalAlignment', 'bottom', ...
                'HitTest', 'off');
            xline(rmsAxes, sample, '-', label, 'Color', color, ...
                'LineWidth', 1.7, 'LabelVerticalAlignment', 'bottom', ...
                'HitTest', 'off');
            sampleIndex = round(sample) + 1;
            if sampleIndex >= 1 && sampleIndex <= currentSignal.sampleCount
                hold(waveformAxes, 'on');
                plot(waveformAxes, sample, currentSignal.filtered(sampleIndex), 'o', ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', ...
                    'MarkerSize', 7, 'HitTest', 'off');
                hold(waveformAxes, 'off');
                hold(rmsAxes, 'on');
                plot(rmsAxes, sample, currentSignal.rmsEnvelope(sampleIndex), 'o', ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', ...
                    'MarkerSize', 7, 'HitTest', 'off');
                hold(rmsAxes, 'off');
            end
        end

        if isfinite(row.main_sample)
            hold(acfAxes, 'on');
            hold(cepstrumAxes, 'on');
            zeroIndex = localLagIndex(currentSignal.acfLags, 0);
            plot(acfAxes, 0, currentSignal.acf(zeroIndex), 'p', ...
                'MarkerFaceColor', colors.main, 'MarkerEdgeColor', 'w', ...
                'MarkerSize', 11, 'HitTest', 'off');
            for markerIndex = 2:3
                sample = row.(markerNames{markerIndex});
                if ~isfinite(sample)
                    continue;
                end
                lag = round(sample - row.main_sample);
                lagIndex = localLagIndex(currentSignal.acfLags, lag);
                if lag > 0 && isfinite(lagIndex)
                    plot(acfAxes, lag, currentSignal.acf(lagIndex), 'o', ...
                        'MarkerFaceColor', markerColors{markerIndex}, ...
                        'MarkerEdgeColor', 'w', 'MarkerSize', 8, 'HitTest', 'off');
                    xline(acfAxes, lag, '--', sprintf('%s: %d', markerLabels{markerIndex}, lag), ...
                        'Color', markerColors{markerIndex}, 'LineWidth', 1.2, ...
                        'HitTest', 'off');
                    quefrencyMs = 1000 * lag / currentSignal.fs;
                    cepstrumIndex = localNearestIndex(currentSignal.quefrencyMs, quefrencyMs);
                    plot(cepstrumAxes, quefrencyMs, ...
                        currentSignal.realCepstrum(cepstrumIndex), 'o', ...
                        'MarkerFaceColor', markerColors{markerIndex}, ...
                        'MarkerEdgeColor', 'w', 'MarkerSize', 8, 'HitTest', 'off');
                    xline(cepstrumAxes, quefrencyMs, '--', ...
                        sprintf('%s: %.3f ms', markerLabels{markerIndex}, quefrencyMs), ...
                        'Color', markerColors{markerIndex}, 'LineWidth', 1.2, ...
                        'HitTest', 'off');
                end
            end
            hold(acfAxes, 'off');
            hold(cepstrumAxes, 'off');
        end
    end

    function drawSpectralEnvelopes(row)
        halfWindowUs = 50;
        displayFloorDb = -80;
        xlabel(spectrumAxes, '频率（kHz）');
        ylabel(spectrumAxes, '相对幅度（dB；主径峰值 = 0 dB）');
        grid(spectrumAxes, 'on');
        xlim(spectrumAxes, [0, currentSignal.fs / 2000]);
        ylim(spectrumAxes, [displayFloorDb, 3]);

        if ~isfinite(row.main_sample)
            title(spectrumAxes, ...
                '频谱包络（选择主径后显示，中心前后各 50 μs）');
            return;
        end

        [frequencyHz, mainEnvelope, windowSamples] = localSpectrumEnvelope( ...
            currentSignal.filtered, currentSignal.fs, row.main_sample, halfWindowUs);
        mainPeak = max(mainEnvelope);
        if ~isfinite(mainPeak) || mainPeak <= eps
            title(spectrumAxes, '频谱包络（主径窗口频谱能量不足）');
            return;
        end

        pathSamples = [row.main_sample, row.multipath1_sample, row.multipath2_sample];
        pathLabels = {'主径', '多径1', '多径2'};
        pathColors = {colors.main, colors.multipath1, colors.multipath2};
        pathWidths = [1.6, 1.35, 1.35];
        labelVerticalAlignments = {'top', 'middle', 'bottom'};
        envelopes = cell(1, 3);
        envelopes{1} = mainEnvelope;
        for pathIndex = 2:3
            if ~isfinite(pathSamples(pathIndex))
                continue;
            end
            [~, envelopes{pathIndex}] = localSpectrumEnvelope(currentSignal.filtered, ...
                currentSignal.fs, pathSamples(pathIndex), halfWindowUs);
        end

        hold(spectrumAxes, 'on');
        highestDb = 0;
        for pathIndex = 1:3
            if isempty(envelopes{pathIndex})
                continue;
            end
            envelopeDb = localAmplitudeToDb(envelopes{pathIndex}, mainPeak, displayFloorDb);
            highestDb = max(highestDb, max(envelopeDb));
            if pathIndex == 1
                correlationSuffix = '';
            else
                coefficient = localPearsonCorrelation(mainEnvelope, envelopes{pathIndex});
                correlationSuffix = sprintf('，r=%.3f', coefficient);
            end

            [lowerHz, upperHz, bandwidthHz, bandwidthValid] = ...
                localMinus3DbBandwidth(frequencyHz, envelopes{pathIndex});
            if bandwidthValid
                bandwidthDescription = sprintf('−3 dB %.1f–%.1f kHz，BW=%.1f kHz', ...
                    lowerHz / 1000, upperHz / 1000, bandwidthHz / 1000);
            else
                bandwidthDescription = '−3 dB 带宽无法确定';
            end
            legendLabel = sprintf('%s（%s%s）', pathLabels{pathIndex}, ...
                bandwidthDescription, correlationSuffix);
            plot(spectrumAxes, frequencyHz / 1000, envelopeDb, ...
                'Color', pathColors{pathIndex}, 'LineWidth', pathWidths(pathIndex), ...
                'DisplayName', legendLabel, 'HitTest', 'off');

            if bandwidthValid
                lowerLine = xline(spectrumAxes, lowerHz / 1000, '--', ...
                    sprintf('%s L %.1f', pathLabels{pathIndex}, lowerHz / 1000), ...
                    'Color', pathColors{pathIndex}, 'LineWidth', 1.0, ...
                    'LabelVerticalAlignment', labelVerticalAlignments{pathIndex}, ...
                    'HitTest', 'off');
                upperLine = xline(spectrumAxes, upperHz / 1000, '--', ...
                    sprintf('%s H %.1f', pathLabels{pathIndex}, upperHz / 1000), ...
                    'Color', pathColors{pathIndex}, 'LineWidth', 1.0, ...
                    'LabelVerticalAlignment', labelVerticalAlignments{pathIndex}, ...
                    'HitTest', 'off');
                lowerLine.Annotation.LegendInformation.IconDisplayStyle = 'off';
                upperLine.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end
        end
        hold(spectrumAxes, 'off');
        upperLimitDb = max(3, 5 * ceil(highestDb / 5));
        ylim(spectrumAxes, [displayFloorDb, upperLimitDb]);
        legend(spectrumAxes, 'show', 'Location', 'northeast');
        title(spectrumAxes, sprintf( ...
            '频谱包络（前后各 50 μs，共 %d 点；共同参考：主径峰值 0 dB）', ...
            windowSamples));
    end

    function drawRelativePhaseSpectrum(row)
        halfWindowUs = 50;
        minimumSpectrumRatio = 0.10;
        minimumFrequencyHz = 20000;
        pathSamples = [row.main_sample, row.multipath1_sample, row.multipath2_sample];
        pathLabels = {'多径1', '多径2'};
        pathColors = {colors.multipath1, colors.multipath2};

        xlabel(phaseAxes, '频率（kHz）');
        ylabel(phaseAxes, '多径相对直达径相位（°）');
        grid(phaseAxes, 'on');
        xlim(phaseAxes, [0, currentSignal.fs / 2000]);
        ylim(phaseAxes, [-190, 190]);
        yticks(phaseAxes, [-180, -90, 0, 90, 180]);
        legend(phaseAxes, 'off');
        if ~isfinite(row.main_sample)
            title(phaseAxes, ...
                '延迟补偿相对相位谱（请先选择主径和多径）');
            return;
        end

        hold(phaseAxes, 'on');
        yline(phaseAxes, 0, ':', '0° 同相', 'Color', [0.35, 0.35, 0.35], ...
            'HandleVisibility', 'off', 'HitTest', 'off');
        yline(phaseAxes, 180, ':', '+180° 反相', 'Color', [0.55, 0.25, 0.25], ...
            'HandleVisibility', 'off', 'HitTest', 'off');
        yline(phaseAxes, -180, ':', '-180° 反相', 'Color', [0.55, 0.25, 0.25], ...
            'HandleVisibility', 'off', 'HitTest', 'off');
        displayedPathCount = 0;
        summaries = strings(0, 1);
        for pathIndex = 2:3
            if ~isfinite(pathSamples(pathIndex))
                continue;
            end
            [frequencyHz, relativePhaseDeg, reliable, meanPhaseDeg, ...
                phaseConcentration, polarityCorrelation] = localRelativePhaseSpectrum( ...
                currentSignal.filtered, currentSignal.fs, pathSamples(1), ...
                pathSamples(pathIndex), halfWindowUs, minimumSpectrumRatio, ...
                minimumFrequencyHz);
            validCount = nnz(reliable);
            verdict = localPhaseVerdict(meanPhaseDeg, phaseConcentration, validCount);
            labelIndex = pathIndex - 1;
            plot(phaseAxes, frequencyHz(reliable) / 1000, ...
                relativePhaseDeg(reliable), '.', 'Color', pathColors{labelIndex}, ...
                'MarkerSize', 9, 'DisplayName', sprintf( ...
                '%s：Δφ=%.0f°，R=%.2f，ρ=%.2f', pathLabels{labelIndex}, ...
                meanPhaseDeg, phaseConcentration, polarityCorrelation), ...
                'HitTest', 'off');
            summaries(end + 1) = sprintf('%s %s', pathLabels{labelIndex}, verdict); %#ok<AGROW>
            displayedPathCount = displayedPathCount + 1;
        end
        hold(phaseAxes, 'off');
        if displayedPathCount == 0
            title(phaseAxes, '延迟补偿相对相位谱（请再选择至少一条多径）');
            return;
        end
        legend(phaseAxes, 'show', 'Location', 'best');
        title(phaseAxes, sprintf('延迟补偿相对相位谱（%s）', strjoin(summaries, '；')));
    end

    function drawAutomaticMarkers()
        if isempty(autoLabels)
            return;
        end
        key = localNormalizePath(fileInfo.relative_path(currentIndex));
        rows = autoLabels(autoLabels.relative_key == key, :);
        if isempty(rows)
            return;
        end
        mainRows = rows(rows.label == "main_arrival", :);
        automaticMain = NaN;
        if ~isempty(mainRows)
            automaticMain = mainRows.center_sample(1);
        end
        for automaticIndex = 1:height(rows)
            sample = rows.center_sample(automaticIndex);
            xline(waveformAxes, sample, ':', '自动', 'Color', colors.automatic, ...
                'LineWidth', 1.0, 'HitTest', 'off');
            xline(rmsAxes, sample, ':', 'Color', colors.automatic, ...
                'LineWidth', 1.0, 'HitTest', 'off');
            if isfinite(automaticMain) && rows.label(automaticIndex) ~= "main_arrival"
                lag = round(sample - automaticMain);
                if lag > 0
                    xline(acfAxes, lag, ':', '自动', 'Color', colors.automatic, ...
                        'LineWidth', 1.0, 'HitTest', 'off');
                    xline(cepstrumAxes, 1000 * lag / currentSignal.fs, ':', ...
                        '自动', 'Color', colors.automatic, 'LineWidth', 1.0, ...
                        'HitTest', 'off');
                end
            end
        end
    end

    function onPlotClicked(mode, axesHandle)
        point = axesHandle.CurrentPoint;
        clickedX = point(1, 1);
        row = annotations(currentIndex, :);

        if strcmp(mode, 'lag') || strcmp(mode, 'quefrency')
            featureName = '自相关图';
            if strcmp(mode, 'quefrency')
                featureName = '实倒谱图';
            end
            if currentTarget == "main"
                setMessage(sprintf('%s只能标注多径；请在波形或局部 RMS 图上标注主径。', ...
                    featureName), true);
                return;
            end
            if ~isfinite(row.main_sample)
                setMessage(sprintf('请先标注主径，再从%s选择多径时延。', featureName), true);
                return;
            end
            if strcmp(mode, 'quefrency')
                delaySamples = round(clickedX * 1e-3 * currentSignal.fs);
            else
                delaySamples = round(clickedX);
            end
            if delaySamples <= 0
                setMessage('零和负时延仅供观察；只能点击正时延标注多径。', true);
                return;
            end
            selectedSample = row.main_sample + delaySamples;
        else
            selectedSample = round(clickedX);
        end

        selectedSample = round(selectedSample);
        if selectedSample < 0 || selectedSample >= currentSignal.sampleCount
            setMessage('点选位置超出当前音频范围。', true);
            return;
        end

        if currentTarget ~= "main"
            if ~isfinite(row.main_sample)
                setMessage('请先标注主径。', true);
                return;
            end
            if selectedSample <= row.main_sample
                setMessage('多径位置必须晚于主径。', true);
                return;
            end
        elseif (isfinite(row.multipath1_sample) && row.multipath1_sample <= selectedSample) || ...
                (isfinite(row.multipath2_sample) && row.multipath2_sample <= selectedSample)
            setMessage('新主径不能位于已有多径之后；请先清除或调整多径。', true);
            return;
        end

        pushUndo();
        switch currentTarget
            case "main"
                annotations.main_sample(currentIndex) = selectedSample;
            case "multipath1"
                if ~isfinite(annotations.multipath1_sample(currentIndex))
                    annotations.multipath1_quality(currentIndex) = "unlabeled";
                end
                annotations.multipath1_sample(currentIndex) = selectedSample;
            case "multipath2"
                if ~isfinite(annotations.multipath2_sample(currentIndex))
                    annotations.multipath2_quality(currentIndex) = "unlabeled";
                end
                annotations.multipath2_sample(currentIndex) = selectedSample;
        end
        annotations = localSortAndDeriveRow(annotations, currentIndex, currentSignal.fs);
        markDirty();
        setMessage(sprintf('已标注 %s：采样点 %d（%.6f ms）', ...
            localTargetDisplayName(currentTarget), selectedSample, ...
            1000 * selectedSample / currentSignal.fs), false);
        refreshUi();
    end

    function onTargetChanged(~, event)
        currentTarget = string(event.NewValue.Tag);
        setMessage(sprintf('当前目标：%s', localTargetDisplayName(currentTarget)), false);
    end

    function onStatusChanged(~, ~)
        pushUndo();
        annotations.status(currentIndex) = string(statusDropdown.Value);
        markDirty();
        refreshUi();
    end

    function onQualityChanged(multipathIndex)
        sampleName = sprintf('multipath%d_sample', multipathIndex);
        qualityName = sprintf('multipath%d_quality', multipathIndex);
        if ~isfinite(annotations.(sampleName)(currentIndex))
            if multipathIndex == 1
                mp1QualityDropdown.Value = 'not_applicable';
            else
                mp2QualityDropdown.Value = 'not_applicable';
            end
            setMessage(sprintf('多径%d尚未标注，不能设置质量。', multipathIndex), true);
            return;
        end
        pushUndo();
        if multipathIndex == 1
            annotations.(qualityName)(currentIndex) = string(mp1QualityDropdown.Value);
        else
            annotations.(qualityName)(currentIndex) = string(mp2QualityDropdown.Value);
        end
        markDirty();
        refreshUi();
    end

    function onNotesChanged(~, ~)
        pushUndo();
        annotations.notes(currentIndex) = string(strjoin(notesArea.Value, newline));
        markDirty();
        refreshUi();
    end

    function onClearCurrent(~, ~)
        variableName = currentTarget + "_sample";
        if ~isfinite(annotations.(variableName)(currentIndex))
            setMessage('当前目标尚未标注。', false);
            return;
        end
        pushUndo();
        annotations.(variableName)(currentIndex) = NaN;
        if currentTarget == "multipath1" || currentTarget == "multipath2"
            annotations.(currentTarget + "_quality")(currentIndex) = "not_applicable";
        end
        annotations = localSortAndDeriveRow(annotations, currentIndex, currentSignal.fs);
        markDirty();
        refreshUi();
    end

    function pushUndo()
        snapshot = annotations(currentIndex, :);
        undoStack{end + 1} = snapshot;
        if numel(undoStack) > 50
            undoStack = undoStack(end - 49:end);
        end
    end

    function onUndo(~, ~)
        if isempty(undoStack)
            return;
        end
        annotations(currentIndex, :) = undoStack{end};
        undoStack(end) = [];
        statusDropdown.Value = char(annotations.status(currentIndex));
        mp1QualityDropdown.Value = char(annotations.multipath1_quality(currentIndex));
        mp2QualityDropdown.Value = char(annotations.multipath2_quality(currentIndex));
        notesArea.Value = cellstr(annotations.notes(currentIndex));
        markDirty();
        refreshUi();
    end

    function markDirty()
        isDirty = ~localRowsEqual(annotations, savedAnnotations);
    end

    function onSaveAndNext(~, ~)
        if ~validateCurrentForFinalSave(true)
            return;
        end
        if saveAnnotations(false) && currentIndex < height(fileInfo)
            currentIndex = currentIndex + 1;
            loadCurrentSample();
        end
    end

    function success = saveAnnotations(showConfirmation)
        success = false;
        synchronizeEditors();
        if ~validateCurrentForFinalSave(false)
            return;
        end
        if annotations.status(currentIndex) ~= "unlabeled"
            annotations.annotated_at(currentIndex) = ...
                string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
        end
        annotations = localSortAndDeriveRow(annotations, currentIndex, currentSignal.fs);
        try
            localAtomicWriteTable(annotations, outputCsv);
        catch exception
            uialert(fig, getReport(exception, 'extended', 'hyperlinks', 'off'), ...
                'CSV 保存失败', 'Icon', 'error');
            return;
        end
        savedAnnotations = annotations;
        isDirty = false;
        success = true;
        setMessage(sprintf('已保存：%s', outputCsv), false);
        refreshUi();
        if showConfirmation
            uialert(fig, sprintf('CSV 已保存至：\n%s', outputCsv), '保存完成', 'Icon', 'success');
        end
    end

    function synchronizeEditors()
        annotations.status(currentIndex) = string(statusDropdown.Value);
        annotations.multipath1_quality(currentIndex) = string(mp1QualityDropdown.Value);
        annotations.multipath2_quality(currentIndex) = string(mp2QualityDropdown.Value);
        annotations.notes(currentIndex) = string(strjoin(notesArea.Value, newline));
    end

    function valid = validateCurrentForFinalSave(requireFinalStatus)
        synchronizeEditors();
        [valid, validationMessage] = localValidateAnnotationRow(annotations(currentIndex, :), ...
            currentSignal.sampleCount, requireFinalStatus);
        if ~valid
            uialert(fig, validationMessage, '标注校验未通过', 'Icon', 'warning');
        end
    end

    function navigateBy(offset)
        destination = currentIndex + offset;
        if destination < 1 || destination > height(fileInfo)
            return;
        end
        synchronizeEditors();
        markDirty();
        currentIndex = destination;
        loadCurrentSample();
    end

    function onSampleSelected(source, event)
        destination = round(double(event.Value));
        if destination == currentIndex
            return;
        end
        if destination < 1 || destination > height(fileInfo)
            source.Value = currentIndex;
            setMessage('所选样本序号超出当前数据集范围。', true);
            return;
        end
        synchronizeEditors();
        markDirty();
        currentIndex = destination;
        loadCurrentSample();
    end

    function proceed = resolveUnsavedChanges()
        synchronizeEditors();
        markDirty();
        proceed = true;
        if ~isDirty
            return;
        end
        choice = uiconfirm(fig, '当前会话有未保存修改。', '未保存修改', ...
            'Options', {'保存', '放弃', '取消'}, ...
            'DefaultOption', 1, 'CancelOption', 3, 'Icon', 'warning');
        switch choice
            case '保存'
                proceed = saveAnnotations(false);
            case '放弃'
                annotations = savedAnnotations;
                isDirty = false;
            otherwise
                proceed = false;
        end
    end

    function onChooseDiagnostics(~, ~)
        if ~resolveUnsavedChanges()
            return;
        end
        selectedDirectory = uigetdir(diagnosticsRoot, '选择 diagnostics 根目录');
        if isequal(selectedDirectory, 0)
            return;
        end
        progressDialog = uiprogressdlg(fig, 'Title', '索引诊断样本', ...
            'Message', '正在验证目录与原始 WAV 映射...', 'Indeterminate', 'on');
        cleanup = onCleanup(@() delete(progressDialog));
        try
            newFileInfo = localBuildDataset(selectedDirectory, audioRoot, true);
            newOutput = fullfile(fileparts(selectedDirectory), ...
                'manual_multipath_annotations.csv');
            newAnnotations = localCreateAnnotationTable(newFileInfo);
            newAnnotations = localLoadExistingAnnotations(newAnnotations, newOutput);
            newAutoLabels = localLoadAutomaticLabels(selectedDirectory);
        catch exception
            uialert(fig, getReport(exception, 'extended', 'hyperlinks', 'off'), ...
                '目录加载失败', 'Icon', 'error');
            return;
        end
        clear cleanup;
        diagnosticsRoot = selectedDirectory;
        outputCsv = newOutput;
        fileInfo = newFileInfo;
        annotations = newAnnotations;
        savedAnnotations = newAnnotations;
        autoLabels = newAutoLabels;
        currentIndex = localFirstIncompleteIndex(annotations);
        signalCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
        isDirty = false;
        sampleSelector.Items = cellstr(localSampleSelectorItems(fileInfo));
        sampleSelector.ItemsData = num2cell((1:height(fileInfo))');
        sampleSelector.Value = currentIndex;
        loadCurrentSample();
    end

    function onExportCsv(~, ~)
        synchronizeEditors();
        [fileName, directory] = uiputfile('*.csv', '导出人工标注 CSV', outputCsv);
        if isequal(fileName, 0)
            return;
        end
        oldOutput = outputCsv;
        outputCsv = fullfile(directory, fileName);
        if ~saveAnnotations(true)
            outputCsv = oldOutput;
        end
    end

    function onKeyPress(~, event)
        modifiers = string(event.Modifier);
        hasControl = any(modifiers == "control");
        switch lower(event.Key)
            case '1'
                targetGroup.SelectedObject = mainRadio;
                currentTarget = "main";
            case '2'
                targetGroup.SelectedObject = mp1Radio;
                currentTarget = "multipath1";
            case '3'
                targetGroup.SelectedObject = mp2Radio;
                currentTarget = "multipath2";
            case 'leftarrow'
                navigateBy(-1);
            case 'rightarrow'
                navigateBy(1);
            case {'backspace', 'delete'}
                onClearCurrent([], []);
            case 's'
                if hasControl
                    saveAnnotations(false);
                end
            case 'z'
                if hasControl
                    onUndo([], []);
                end
        end
        refreshUi();
    end

    function onCloseRequested(~, ~)
        if isClosing
            delete(fig);
            return;
        end
        if ~resolveUnsavedChanges()
            return;
        end
        isClosing = true;
        delete(fig);
    end

    function setMessage(message, isWarning)
        messageLabel.Text = message;
        if isWarning
            messageLabel.FontColor = [0.78, 0.20, 0.10];
        else
            messageLabel.FontColor = [0.15, 0.35, 0.18];
        end
    end
end

function dataset = localBuildDataset(diagnosticsRoot, audioRoot, showProgress)
    diagnosticsRoot = char(string(diagnosticsRoot));
    audioRoot = char(string(audioRoot));
    if ~isfolder(diagnosticsRoot)
        error('ManualMultipathAnnotator:MissingDiagnostics', ...
            'Diagnostics 目录不存在：%s', diagnosticsRoot);
    end
    if ~isfolder(audioRoot)
        error('ManualMultipathAnnotator:MissingAudioRoot', ...
            '原始音频目录不存在：%s', audioRoot);
    end

    diagnosticPlots = dir(fullfile(diagnosticsRoot, '**', 'filter_and_detection.png'));
    if isempty(diagnosticPlots)
        error('ManualMultipathAnnotator:NoDiagnostics', ...
            '未在目录中找到 filter_and_detection.png：%s', diagnosticsRoot);
    end
    diagnosticFolders = unique(string({diagnosticPlots.folder}'), 'stable');
    count = numel(diagnosticFolders);
    sourcePath = strings(count, 1);
    relativePath = strings(count, 1);
    voyage = strings(count, 1);
    pulseTrain = strings(count, 1);
    clickName = strings(count, 1);
    diagnosticDir = strings(count, 1);
    prefix = string(diagnosticsRoot) + filesep;

    if showProgress
        fprintf('总体进度: [%-30s] %3d%%  索引 diagnostics\r', '', 0);
    end
    for index = 1:count
        folder = diagnosticFolders(index);
        if ~startsWith(folder, prefix, 'IgnoreCase', true)
            error('ManualMultipathAnnotator:InvalidDiagnosticPath', ...
                '诊断路径不位于根目录下：%s', folder);
        end
        relativeFolder = extractAfter(folder, strlength(prefix));
        parts = split(relativeFolder, filesep);
        if numel(parts) ~= 3
            error('ManualMultipathAnnotator:UnexpectedHierarchy', ...
                '预期 diagnostics/航次/脉冲串/Click 三级结构，实际为：%s', folder);
        end
        partCells = cellstr(parts);
        mappedDirectory = fullfile(audioRoot, partCells{:});
        wavFiles = dir(fullfile(mappedDirectory, '*.wav'));
        if numel(wavFiles) ~= 1
            error('ManualMultipathAnnotator:AmbiguousAudioMapping', ...
                '目录必须恰好包含一个 WAV，实际为 %d 个：%s', ...
                numel(wavFiles), mappedDirectory);
        end
        sourcePath(index) = string(fullfile(wavFiles(1).folder, wavFiles(1).name));
        relativePath(index) = string(fullfile(partCells{:}, wavFiles(1).name));
        voyage(index) = parts(1);
        pulseTrain(index) = parts(2);
        clickName(index) = parts(3);
        diagnosticDir(index) = folder;

        if showProgress && (index == count || mod(index, max(1, floor(count / 50))) == 0)
            percentage = round(100 * index / count);
            filled = round(30 * index / count);
            fprintf('总体进度: [%s%s] %3d%%  索引 diagnostics\r', ...
                repmat('#', 1, filled), repmat('-', 1, 30 - filled), percentage);
        end
    end
    if showProgress
        fprintf('\n');
    end

    dataset = table(sourcePath, relativePath, voyage, pulseTrain, clickName, ...
        diagnosticDir, 'VariableNames', {'source_path', 'relative_path', ...
        'voyage', 'pulse_train', 'click', 'diagnostic_dir'});
    dataset = sortrows(dataset, 'relative_path');
end

function annotations = localCreateAnnotationTable(fileInfo)
    count = height(fileInfo);
    [sampleRate, sampleCount] = localInspectAudioHeaders(fileInfo.source_path);
    annotations = table(fileInfo.source_path, fileInfo.relative_path, ...
        fileInfo.voyage, fileInfo.pulse_train, fileInfo.click, ...
        sampleRate, sampleCount, repmat("unlabeled", count, 1), ...
        nan(count, 1), nan(count, 1), nan(count, 1), nan(count, 1), ...
        nan(count, 1), nan(count, 1), repmat("not_applicable", count, 1), ...
        nan(count, 1), nan(count, 1), nan(count, 1), nan(count, 1), ...
        repmat("not_applicable", count, 1), repmat("", count, 1), ...
        repmat("", count, 1), ...
        'VariableNames', {'source_path', 'relative_path', 'voyage', ...
        'pulse_train', 'click', 'sample_rate', 'sample_count', 'status', ...
        'main_sample', 'main_time_ms', 'multipath1_sample', 'multipath1_time_ms', ...
        'multipath1_delay_samples', 'multipath1_delay_ms', 'multipath1_quality', ...
        'multipath2_sample', ...
        'multipath2_time_ms', 'multipath2_delay_samples', 'multipath2_delay_ms', ...
        'multipath2_quality', 'notes', 'annotated_at'});
end

function [sampleRate, sampleCount] = localInspectAudioHeaders(sourcePaths)
    count = numel(sourcePaths);
    sampleRate = zeros(count, 1);
    sampleCount = zeros(count, 1);
    for index = 1:count
        info = audioinfo(char(sourcePaths(index)));
        sampleRate(index) = info.SampleRate;
        sampleCount(index) = info.TotalSamples;
    end
end

function annotations = localLoadExistingAnnotations(annotations, outputCsv)
    if ~isfile(outputCsv)
        return;
    end
    existing = readtable(outputCsv, 'TextType', 'string', ...
        'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
    qualityNames = {'multipath1_quality', 'multipath2_quality'};
    required = setdiff(annotations.Properties.VariableNames, qualityNames, 'stable');
    missing = setdiff(required, existing.Properties.VariableNames, 'stable');
    if ~isempty(missing)
        error('ManualMultipathAnnotator:InvalidExistingCsv', ...
            '已有 CSV 缺少字段：%s', strjoin(missing, ', '));
    end

    existingKey = localNormalizePath(existing.relative_path);
    if numel(unique(existingKey)) ~= numel(existingKey)
        error('ManualMultipathAnnotator:DuplicateCsvRows', ...
            '已有 CSV 的 relative_path 存在重复行：%s', outputCsv);
    end
    annotationKey = localNormalizePath(annotations.relative_path);
    [matched, locations] = ismember(annotationKey, existingKey);
    numericNames = {'sample_rate', 'sample_count', 'main_sample', 'main_time_ms', ...
        'multipath1_sample', 'multipath1_time_ms', 'multipath1_delay_samples', ...
        'multipath1_delay_ms', 'multipath2_sample', 'multipath2_time_ms', ...
        'multipath2_delay_samples', 'multipath2_delay_ms'};
    stringNames = setdiff(required, numericNames, 'stable');

    for rowIndex = find(matched)'
        sourceIndex = locations(rowIndex);
        for nameIndex = 1:numel(numericNames)
            name = numericNames{nameIndex};
            annotations.(name)(rowIndex) = localScalarDouble(existing.(name)(sourceIndex));
        end
        for nameIndex = 1:numel(stringNames)
            name = stringNames{nameIndex};
            annotations.(name)(rowIndex) = string(existing.(name)(sourceIndex));
        end
        for multipathIndex = 1:2
            sampleName = sprintf('multipath%d_sample', multipathIndex);
            delayName = sprintf('multipath%d_delay_ms', multipathIndex);
            qualityName = sprintf('multipath%d_quality', multipathIndex);
            quality = "";
            if ismember(qualityName, existing.Properties.VariableNames)
                quality = string(existing.(qualityName)(sourceIndex));
            end
            if ~isfinite(annotations.(sampleName)(rowIndex))
                quality = "not_applicable";
            elseif ismissing(quality) || strlength(strtrim(quality)) == 0 || ...
                    quality == "not_applicable"
                if annotations.(delayName)(rowIndex) > 1.5
                    quality = "exclude_late";
                else
                    quality = "unlabeled";
                end
            end
            annotations.(qualityName)(rowIndex) = quality;
        end
    end

    validStatuses = ["unlabeled", "completed", "no_multipath", "uncertain", "skipped"];
    invalid = ~ismember(annotations.status, validStatuses);
    if any(invalid)
        error('ManualMultipathAnnotator:InvalidStatus', ...
            '已有 CSV 包含不支持的状态：%s', strjoin(unique(annotations.status(invalid)), ', '));
    end
    validQualities = ["not_applicable", "unlabeled", "clean", "overlapping", ...
        "reverberant", "uncertain", "exclude_late"];
    for qualityName = string(qualityNames)
        invalidQuality = ~ismember(annotations.(qualityName), validQualities);
        if any(invalidQuality)
            error('ManualMultipathAnnotator:InvalidQuality', ...
                '已有 CSV 的 %s 包含不支持的质量标签：%s', qualityName, ...
                strjoin(unique(annotations.(qualityName)(invalidQuality)), ', '));
        end
    end
end

function autoLabels = localLoadAutomaticLabels(diagnosticsRoot)
    automaticCsv = fullfile(fileparts(diagnosticsRoot), 'selected_paths.csv');
    if ~isfile(automaticCsv)
        autoLabels = table();
        return;
    end
    tableData = readtable(automaticCsv, 'TextType', 'string', ...
        'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
    required = {'relative_path', 'label', 'center_sample'};
    if ~all(ismember(required, tableData.Properties.VariableNames))
        warning('ManualMultipathAnnotator:InvalidAutomaticCsv', ...
            '自动结果缺少必要字段，已禁用自动参考：%s', automaticCsv);
        autoLabels = table();
        return;
    end
    centerSample = localNumericColumn(tableData.center_sample);
    relativeKey = localNormalizePath(tableData.relative_path);
    autoLabels = table(relativeKey, string(tableData.label), centerSample, ...
        'VariableNames', {'relative_key', 'label', 'center_sample'});
end

function signal = localPrepareSignal(sourcePath)
    [raw, sampleRate] = audioread(sourcePath);
    if size(raw, 2) > 1
        raw = mean(raw, 2);
    end
    raw = double(raw(:));
    if numel(raw) < 64
        error('ManualMultipathAnnotator:ShortAudio', ...
            'WAV 过短，仅有 %d 个采样点：%s', numel(raw), sourcePath);
    end
    if any(~isfinite(raw))
        error('ManualMultipathAnnotator:InvalidAudio', 'WAV 包含 NaN 或 Inf：%s', sourcePath);
    end
    raw = raw - mean(raw);
    peak = max(abs(raw));
    if peak <= eps
        error('ManualMultipathAnnotator:SilentAudio', 'WAV 接近全零：%s', sourcePath);
    end
    raw = raw / peak;
    filtered = localFilterSignal(raw, sampleRate);
    rmsWindowUs = 20;
    [rmsEnvelope, rmsWindowSamples] = localRmsEnergy(filtered, sampleRate, rmsWindowUs);
    [acf, acfLags] = localCenteredAutocorrelation(filtered);
    [realCepstrum, quefrencyMs] = localRealCepstrum(filtered, sampleRate);
    signal = struct('raw', raw, 'filtered', filtered, ...
        'rmsEnvelope', rmsEnvelope, 'rmsWindowSamples', rmsWindowSamples, ...
        'acf', acf, 'acfLags', acfLags, 'fs', sampleRate, ...
        'realCepstrum', realCepstrum, 'quefrencyMs', quefrencyMs, ...
        'sampleCount', numel(filtered));
end

function filtered = localFilterSignal(signal, sampleRate)
    nyquist = sampleRate / 2;
    if nyquist <= 215000
        error('ManualMultipathAnnotator:InsufficientSampleRate', ...
            '采样率 %d Hz 的 Nyquist 频率不足以设计 185–215 kHz 带阻。', sampleRate);
    end
    [zHigh, pHigh, kHigh] = butter(2, 20000 / nyquist, 'high');
    [sosHigh, gainHigh] = zp2sos(zHigh, pHigh, kHigh);
    warningState = warning('off', 'signal:filtfilt:ParseSOS');
    warningCleanup = onCleanup(@() warning(warningState));
    filtered = filtfilt(sosHigh, gainHigh, signal);
    [zStop, pStop, kStop] = butter(2, [185000, 215000] / nyquist, 'stop');
    [sosStop, gainStop] = zp2sos(zStop, pStop, kStop);
    filtered = filtfilt(sosStop, gainStop, filtered);
    clear warningCleanup;
    filtered = filtered - mean(filtered);
    peak = max(abs(filtered));
    if peak <= eps
        error('ManualMultipathAnnotator:ZeroFilteredSignal', '滤波后信号接近全零。');
    end
    filtered = filtered / peak;
end

function [rmsEnvelope, windowSamples] = localRmsEnergy(signal, sampleRate, windowUs)
    signal = signal(:);
    windowSamples = max(1, round(windowUs * 1e-6 * sampleRate));
    if mod(windowSamples, 2) == 0
        windowSamples = windowSamples + 1;
    end
    rmsEnvelope = sqrt(movmean(signal.^2, windowSamples, 'Endpoints', 'shrink'));
    peak = max(rmsEnvelope);
    if peak > 0
        rmsEnvelope = rmsEnvelope / peak;
    end
end

function [frequencyHz, envelope, windowSamples] = localSpectrumEnvelope( ...
        signal, sampleRate, centerSample, halfWindowUs)
    signal = signal(:);
    halfWindowSamples = max(1, round(halfWindowUs * 1e-6 * sampleRate));
    windowSamples = 2 * halfWindowSamples + 1;
    centerSample = round(centerSample);
    if centerSample < 0 || centerSample >= numel(signal)
        error('ManualMultipathAnnotator:SpectrumCenterOutOfRange', ...
            '频谱中心采样点 %d 超出 0 到 %d 的范围。', centerSample, numel(signal) - 1);
    end

    segment = zeros(windowSamples, 1);
    requestedStart = centerSample - halfWindowSamples;
    requestedEnd = centerSample + halfWindowSamples;
    sourceStart = max(0, requestedStart);
    sourceEnd = min(numel(signal) - 1, requestedEnd);
    destinationStart = sourceStart - requestedStart + 1;
    destinationEnd = destinationStart + sourceEnd - sourceStart;
    segment(destinationStart:destinationEnd) = signal((sourceStart:sourceEnd) + 1);

    tapered = segment .* hann(windowSamples, 'periodic');
    fftLength = max(256, 2 ^ nextpow2(windowSamples));
    spectrum = abs(fft(tapered, fftLength));
    oneSidedCount = floor(fftLength / 2) + 1;
    envelope = spectrum(1:oneSidedCount);
    frequencyHz = (0:(oneSidedCount - 1))' * (sampleRate / fftLength);
end

function [frequencyHz, relativePhaseDeg, reliable, meanPhaseDeg, ...
        phaseConcentration, polarityCorrelation] = localRelativePhaseSpectrum( ...
        signal, sampleRate, mainSample, multipathSample, halfWindowUs, ...
        minimumSpectrumRatio, minimumFrequencyHz)
    signal = signal(:);
    halfWindowSamples = max(1, round(halfWindowUs * 1e-6 * sampleRate));
    windowSamples = 2 * halfWindowSamples + 1;
    mainSample = round(mainSample);
    multipathSample = round(multipathSample);
    if any([mainSample, multipathSample] < 0) || ...
            any([mainSample, multipathSample] >= numel(signal))
        error('ManualMultipathAnnotator:RelativePhaseCenterOutOfRange', ...
            '相对相位谱的路径中心超出 0 到 %d 的范围。', numel(signal) - 1);
    end

    mainSegment = localCenteredSegment(signal, mainSample, halfWindowSamples);
    multipathSegment = localCenteredSegment(signal, multipathSample, halfWindowSamples);
    taper = hann(windowSamples, 'periodic');
    mainTapered = mainSegment .* taper;
    multipathTapered = multipathSegment .* taper;
    fftLength = max(256, 2 ^ nextpow2(windowSamples));
    mainSpectrum = fft(mainTapered, fftLength);
    multipathSpectrum = fft(multipathTapered, fftLength);
    oneSidedCount = floor(fftLength / 2) + 1;
    mainSpectrum = mainSpectrum(1:oneSidedCount);
    multipathSpectrum = multipathSpectrum(1:oneSidedCount);
    frequencyHz = (0:(oneSidedCount - 1))' * (sampleRate / fftLength);
    crossSpectrum = multipathSpectrum .* conj(mainSpectrum);
    relativePhaseDeg = rad2deg(angle(crossSpectrum));

    mainMagnitude = abs(mainSpectrum);
    multipathMagnitude = abs(multipathSpectrum);
    reliable = frequencyHz >= minimumFrequencyHz & ...
        mainMagnitude >= minimumSpectrumRatio * max(mainMagnitude) & ...
        multipathMagnitude >= minimumSpectrumRatio * max(multipathMagnitude);
    weights = abs(crossSpectrum(reliable));
    if isempty(weights) || sum(weights) <= 0
        meanPhaseDeg = NaN;
        phaseConcentration = 0;
    else
        phaseVector = sum(weights .* exp(1i * deg2rad(relativePhaseDeg(reliable))));
        meanPhaseDeg = rad2deg(angle(phaseVector));
        phaseConcentration = abs(phaseVector) / sum(weights);
    end

    denominator = norm(mainTapered) * norm(multipathTapered);
    if denominator <= eps
        polarityCorrelation = NaN;
    else
        polarityCorrelation = dot(mainTapered, multipathTapered) / denominator;
    end
end

function segment = localCenteredSegment(signal, centerSample, halfWindowSamples)
    windowSamples = 2 * halfWindowSamples + 1;
    segment = zeros(windowSamples, 1);
    requestedStart = centerSample - halfWindowSamples;
    requestedEnd = centerSample + halfWindowSamples;
    sourceStart = max(0, requestedStart);
    sourceEnd = min(numel(signal) - 1, requestedEnd);
    destinationStart = sourceStart - requestedStart + 1;
    destinationEnd = destinationStart + sourceEnd - sourceStart;
    segment(destinationStart:destinationEnd) = signal((sourceStart:sourceEnd) + 1);
end

function verdict = localPhaseVerdict(meanPhaseDeg, concentration, validCount)
    if ~isfinite(meanPhaseDeg) || concentration < 0.5 || validCount < 3
        verdict = '证据不足';
        return;
    end
    distanceToInPhase = abs(meanPhaseDeg);
    distanceToOppositePhase = abs(180 - abs(meanPhaseDeg));
    if distanceToOppositePhase <= 45
        verdict = '倾向反相（海面候选）';
    elseif distanceToInPhase <= 45
        verdict = '倾向同相';
    else
        verdict = '相位混合';
    end
end

function amplitudeDb = localAmplitudeToDb(amplitude, referenceAmplitude, floorDb)
    if ~isscalar(referenceAmplitude) || ~isfinite(referenceAmplitude) || ...
            referenceAmplitude <= 0
        error('ManualMultipathAnnotator:InvalidSpectrumReference', ...
            'dB 频谱的参考幅度必须是有限正数。');
    end
    amplitude = max(0, double(amplitude));
    minimumRatio = 10 ^ (floorDb / 20);
    amplitudeDb = 20 * log10(max(amplitude / referenceAmplitude, minimumRatio));
end

function [lowerHz, upperHz, bandwidthHz, valid] = ...
        localMinus3DbBandwidth(frequencyHz, amplitude)
    lowerHz = NaN;
    upperHz = NaN;
    bandwidthHz = NaN;
    valid = false;
    frequencyHz = double(frequencyHz(:));
    amplitude = double(amplitude(:));
    if numel(frequencyHz) ~= numel(amplitude) || numel(amplitude) < 3 || ...
            any(~isfinite(frequencyHz)) || any(diff(frequencyHz) <= 0) || ...
            any(~isfinite(amplitude)) || any(amplitude < 0)
        return;
    end

    [peakAmplitude, peakIndex] = max(amplitude);
    if peakAmplitude <= 0 || peakIndex == 1 || peakIndex == numel(amplitude)
        return;
    end
    thresholdDb = -3;
    relativeDb = 20 * log10(max(amplitude / peakAmplitude, realmin('double')));
    leftIndex = find(relativeDb(1:(peakIndex - 1)) <= thresholdDb, 1, 'last');
    rightOffset = find(relativeDb((peakIndex + 1):end) <= thresholdDb, 1, 'first');
    if isempty(leftIndex) || isempty(rightOffset)
        return;
    end
    rightIndex = peakIndex + rightOffset;

    lowerHz = localInterpolateThresholdCrossing( ...
        frequencyHz(leftIndex), relativeDb(leftIndex), ...
        frequencyHz(leftIndex + 1), relativeDb(leftIndex + 1), thresholdDb);
    upperHz = localInterpolateThresholdCrossing( ...
        frequencyHz(rightIndex - 1), relativeDb(rightIndex - 1), ...
        frequencyHz(rightIndex), relativeDb(rightIndex), thresholdDb);
    if ~isfinite(lowerHz) || ~isfinite(upperHz) || upperHz <= lowerHz
        lowerHz = NaN;
        upperHz = NaN;
        return;
    end
    bandwidthHz = upperHz - lowerHz;
    valid = true;
end

function crossingX = localInterpolateThresholdCrossing(x1, y1, x2, y2, threshold)
    if y1 == threshold
        crossingX = x1;
    elseif y2 == threshold
        crossingX = x2;
    elseif y1 == y2
        crossingX = NaN;
    else
        crossingX = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1);
    end
end

function coefficient = localPearsonCorrelation(first, second)
    first = first(:) - mean(first(:));
    second = second(:) - mean(second(:));
    denominator = norm(first) * norm(second);
    if denominator <= eps
        coefficient = NaN;
        return;
    end
    coefficient = dot(first, second) / denominator;
    coefficient = max(-1, min(1, coefficient));
end

function [acf, lags] = localCenteredAutocorrelation(signal)
    signal = signal(:);
    acf = xcorr(signal, signal, 'none');
    lags = (-(numel(signal) - 1):(numel(signal) - 1))';
    zeroLag = acf(numel(signal));
    if zeroLag > 0
        acf = acf / zeroLag;
    else
        acf = zeros(size(acf));
        acf(numel(signal)) = 1;
    end
end

function annotations = localSortAndDeriveRow(annotations, rowIndex, sampleRate)
    multipaths = [annotations.multipath1_sample(rowIndex), ...
        annotations.multipath2_sample(rowIndex)];
    qualities = [annotations.multipath1_quality(rowIndex), ...
        annotations.multipath2_quality(rowIndex)];
    finiteIndexes = find(isfinite(multipaths));
    [finiteValues, order] = sort(multipaths(finiteIndexes));
    finiteQualities = qualities(finiteIndexes(order));
    multipaths(:) = NaN;
    qualities(:) = "not_applicable";
    multipaths(1:numel(finiteValues)) = finiteValues;
    qualities(1:numel(finiteQualities)) = finiteQualities;
    annotations.multipath1_sample(rowIndex) = multipaths(1);
    annotations.multipath2_sample(rowIndex) = multipaths(2);
    annotations.multipath1_quality(rowIndex) = qualities(1);
    annotations.multipath2_quality(rowIndex) = qualities(2);

    mainSample = annotations.main_sample(rowIndex);
    annotations.main_time_ms(rowIndex) = localSampleTime(mainSample, sampleRate);
    for multipathIndex = 1:2
        sampleName = sprintf('multipath%d_sample', multipathIndex);
        timeName = sprintf('multipath%d_time_ms', multipathIndex);
        delaySampleName = sprintf('multipath%d_delay_samples', multipathIndex);
        delayTimeName = sprintf('multipath%d_delay_ms', multipathIndex);
        sample = annotations.(sampleName)(rowIndex);
        annotations.(timeName)(rowIndex) = localSampleTime(sample, sampleRate);
        if isfinite(sample) && isfinite(mainSample)
            delaySamples = sample - mainSample;
            annotations.(delaySampleName)(rowIndex) = delaySamples;
            annotations.(delayTimeName)(rowIndex) = 1000 * delaySamples / sampleRate;
        else
            annotations.(delaySampleName)(rowIndex) = NaN;
            annotations.(delayTimeName)(rowIndex) = NaN;
        end
    end
end

function timeMs = localSampleTime(sample, sampleRate)
    if isfinite(sample)
        timeMs = 1000 * sample / sampleRate;
    else
        timeMs = NaN;
    end
end

function [valid, message] = localValidateAnnotationRow(row, sampleCount, requireFinalStatus)
    valid = false;
    message = '';
    status = string(row.status);
    supported = ["unlabeled", "completed", "no_multipath", "uncertain", "skipped"];
    if ~ismember(status, supported)
        message = sprintf('不支持的状态：%s', status);
        return;
    end
    if requireFinalStatus && status == "unlabeled"
        message = '“保存并下一条”要求先选择已完成、无多径、不确定或跳过。';
        return;
    end

    samples = [row.main_sample, row.multipath1_sample, row.multipath2_sample];
    finiteSamples = samples(isfinite(samples));
    if any(finiteSamples < 0 | finiteSamples >= sampleCount | finiteSamples ~= round(finiteSamples))
        message = sprintf('所有采样点必须是 0 到 %d 之间的整数。', sampleCount - 1);
        return;
    end
    if isfinite(row.multipath1_sample) && (~isfinite(row.main_sample) || ...
            row.multipath1_sample <= row.main_sample)
        message = '多径1必须位于主径之后。';
        return;
    end
    if isfinite(row.multipath2_sample) && (~isfinite(row.main_sample) || ...
            row.multipath2_sample <= row.main_sample)
        message = '多径2必须位于主径之后。';
        return;
    end
    if isfinite(row.multipath1_sample) && isfinite(row.multipath2_sample) && ...
            row.multipath1_sample == row.multipath2_sample
        message = '两条多径不能位于同一个采样点。';
        return;
    end

    supportedQualities = ["not_applicable", "unlabeled", "clean", "overlapping", ...
        "reverberant", "uncertain", "exclude_late"];
    for multipathIndex = 1:2
        sampleName = sprintf('multipath%d_sample', multipathIndex);
        qualityName = sprintf('multipath%d_quality', multipathIndex);
        delayName = sprintf('multipath%d_delay_ms', multipathIndex);
        quality = string(row.(qualityName));
        if ~ismember(quality, supportedQualities)
            message = sprintf('多径%d包含不支持的质量标签：%s。', multipathIndex, quality);
            return;
        end
        if isfinite(row.(sampleName))
            if quality == "not_applicable"
                message = sprintf('多径%d已有位置，质量不能为“不适用”。', multipathIndex);
                return;
            end
            if requireFinalStatus && quality == "unlabeled"
                message = sprintf('请先补充多径%d质量，再保存并进入下一条。', multipathIndex);
                return;
            end
            if row.(delayName) > 1.5 && quality ~= "exclude_late"
                message = sprintf('多径%d时延超过1.5 ms，请标为“1.5 ms后排除”。', multipathIndex);
                return;
            end
            if quality == "exclude_late" && row.(delayName) <= 1.5
                message = sprintf('多径%d时延未超过1.5 ms，不能标为“1.5 ms后排除”。', multipathIndex);
                return;
            end
        elseif quality ~= "not_applicable"
            message = sprintf('多径%d没有位置，质量必须为“不适用”。', multipathIndex);
            return;
        end
    end

    switch status
        case "completed"
            if ~isfinite(row.main_sample) || ...
                    ~(isfinite(row.multipath1_sample) || isfinite(row.multipath2_sample))
                message = '“已完成”要求一个主径和至少一个多径。';
                return;
            end
        case "no_multipath"
            if ~isfinite(row.main_sample)
                message = '“无可用多径”仍要求标注主径。';
                return;
            end
            if isfinite(row.multipath1_sample) || isfinite(row.multipath2_sample)
                message = '“无可用多径”状态下，多径位置必须为空。';
                return;
            end
    end
    valid = true;
end

function localAtomicWriteTable(tableData, outputCsv)
    outputDirectory = fileparts(outputCsv);
    if isempty(outputDirectory)
        outputDirectory = pwd;
        outputCsv = fullfile(outputDirectory, outputCsv);
    end
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    temporaryPath = [tempname(outputDirectory), '.csv'];
    cleanup = onCleanup(@() localDeleteIfExists(temporaryPath));
    writetable(tableData, temporaryPath, 'Encoding', 'UTF-8');
    localEnsureUtf8Bom(temporaryPath);

    backupPath = [outputCsv, '.bak'];
    if isfile(outputCsv)
        [copied, copyMessage] = copyfile(outputCsv, backupPath, 'f');
        if ~copied
            error('ManualMultipathAnnotator:BackupFailed', ...
                '无法创建 CSV 备份：%s', copyMessage);
        end
    end
    [moved, moveMessage] = movefile(temporaryPath, outputCsv, 'f');
    if ~moved
        error('ManualMultipathAnnotator:AtomicSaveFailed', ...
            '无法替换目标 CSV：%s', moveMessage);
    end
    clear cleanup;
end

function localEnsureUtf8Bom(path)
    readHandle = fopen(path, 'rb');
    if readHandle < 0
        error('ManualMultipathAnnotator:CsvReadFailed', '无法读取临时 CSV：%s', path);
    end
    readCleanup = onCleanup(@() fclose(readHandle));
    bytes = fread(readHandle, Inf, '*uint8');
    clear readCleanup;
    if numel(bytes) >= 3 && isequal(bytes(1:3)', uint8([239, 187, 191]))
        return;
    end
    writeHandle = fopen(path, 'wb');
    if writeHandle < 0
        error('ManualMultipathAnnotator:CsvWriteFailed', '无法写入临时 CSV：%s', path);
    end
    writeCleanup = onCleanup(@() fclose(writeHandle));
    fwrite(writeHandle, uint8([239, 187, 191]), 'uint8');
    fwrite(writeHandle, bytes, 'uint8');
    clear writeCleanup;
end

function report = localRunSelfTests(diagnosticsRoot, audioRoot)
    tests = strings(8, 1);
    fprintf('总体进度: [%-30s] %3d%%  MATLAB GUI 自检\r', '', 0);

    syntheticRate = 576000;
    sampleCount = 5761;
    index = (0:(sampleCount - 1))';
    synthetic = exp(-0.5 * ((index - 2880) / 10).^2) .* ...
        sin(2 * pi * 90000 * index / syntheticRate);
    synthetic = synthetic + 0.45 * exp(-0.5 * ((index - 3110) / 10).^2) .* ...
        sin(2 * pi * 90000 * (index - 230) / syntheticRate);
    filtered = localFilterSignal(synthetic, syntheticRate);
    assert(numel(filtered) == sampleCount && all(isfinite(filtered)));
    tests(1) = "filter_length_and_finite";
    localPrintTestProgress(1, 8);

    rmsInput = zeros(101, 1);
    rmsInput(51) = 2;
    [rmsEnvelope, rmsWindowSamples] = localRmsEnergy(rmsInput, syntheticRate, 20);
    assert(rmsWindowSamples == 13 && numel(rmsEnvelope) == numel(rmsInput));
    assert(all(isfinite(rmsEnvelope)) && abs(max(rmsEnvelope) - 1) < 1e-12);
    assert(rmsEnvelope(51) == 1 && rmsEnvelope(1) == 0);
    spectralInput = sin(2 * pi * 90000 * index / syntheticRate);
    [frequencyHz, spectralEnvelope, spectralWindowSamples] = ...
        localSpectrumEnvelope(spectralInput, syntheticRate, 2880, 50);
    [~, halfAmplitudeEnvelope] = localSpectrumEnvelope( ...
        0.5 * spectralInput, syntheticRate, 2880, 50);
    [boundaryFrequencyHz, boundaryEnvelope] = ...
        localSpectrumEnvelope(spectralInput, syntheticRate, 0, 50);
    [~, differentEnvelope] = localSpectrumEnvelope( ...
        sin(2 * pi * 140000 * index / syntheticRate), syntheticRate, 2880, 50);
    [~, peakIndex] = max(spectralEnvelope);
    assert(spectralWindowSamples == 59 && numel(frequencyHz) == 129);
    assert(max(spectralEnvelope) > 1 && ...
        abs(frequencyHz(peakIndex) - 90000) <= syntheticRate / 256);
    assert(abs(max(halfAmplitudeEnvelope) / max(spectralEnvelope) - 0.5) < 1e-12);
    mainDb = localAmplitudeToDb(spectralEnvelope, max(spectralEnvelope), -80);
    halfAmplitudeDb = localAmplitudeToDb( ...
        halfAmplitudeEnvelope, max(spectralEnvelope), -80);
    assert(abs(max(mainDb)) < 1e-12 && ...
        abs(max(halfAmplitudeDb) + 6.020599913279624) < 1e-10);
    assert(isequal(frequencyHz, boundaryFrequencyHz) && ...
        all(isfinite(boundaryEnvelope)));
    assert(abs(localPearsonCorrelation(spectralEnvelope, spectralEnvelope) - 1) < 1e-12);
    assert(abs(localPearsonCorrelation(spectralEnvelope, halfAmplitudeEnvelope) - 1) < 1e-12);
    assert(localPearsonCorrelation(spectralEnvelope, differentEnvelope) < 0.5);
    bandFrequencyHz = (0:10)' * 1000;
    multiPeakAmplitude = [0.1; 0.4; 0.8; 1.0; 0.8; 0.4; 0.1; 0.9; 0.8; 0.1; 0.01];
    [lowerHz, upperHz, bandwidthHz, bandwidthValid] = ...
        localMinus3DbBandwidth(bandFrequencyHz, multiPeakAmplitude);
    assert(bandwidthValid && lowerHz > 1000 && lowerHz < 2000 && ...
        upperHz > 4000 && upperHz < 5000 && abs(bandwidthHz - (upperHz - lowerHz)) < eps);
    [~, ~, ~, missingCrossingValid] = localMinus3DbBandwidth( ...
        [0; 1000; 2000], [0.8; 1.0; 0.9]);
    assert(~missingCrossingValid);
    tests(2) = "spectral_db_relative_level_and_minus3db_bandwidth";
    localPrintTestProgress(2, 8);

    [acf, lags] = localCenteredAutocorrelation(filtered);
    zeroIndex = find(lags == 0, 1, 'first');
    assert(numel(acf) == 2 * sampleCount - 1 && zeroIndex == sampleCount);
    assert(abs(acf(zeroIndex) - 1) < 1e-12 && ...
        max(abs(acf - flipud(acf))) < 1e-10);
    [realCepstrum, quefrencyMs] = localRealCepstrum(filtered, syntheticRate);
    cepstrumZeroIndex = find(quefrencyMs == 0, 1, 'first');
    assert(numel(realCepstrum) == sampleCount && cepstrumZeroIndex == 2881);
    assert(all(isfinite(realCepstrum)) && ...
        max(abs(realCepstrum - flipud(realCepstrum))) < 1e-10);
    oppositeSignal = zeros(sampleCount, 1);
    pulse = exp(-0.5 * (((-29:29)') / 10).^2) .* ...
        sin(2 * pi * 90000 * ((-29:29)') / syntheticRate);
    oppositeSignal((2880 - 29:2880 + 29) + 1) = pulse;
    oppositeSignal((3110 - 29:3110 + 29) + 1) = -0.45 * pulse;
    [relativeFrequencyHz, relativePhaseDeg, reliablePhase, meanPhaseDeg, ...
        phaseConcentration, polarityCorrelation] = localRelativePhaseSpectrum( ...
        oppositeSignal, syntheticRate, 2880, 3110, 50, 0.10, 20000);
    assert(numel(relativeFrequencyHz) == 129 && nnz(reliablePhase) >= 3);
    assert(all(isfinite(relativePhaseDeg)) && ...
        abs(180 - abs(meanPhaseDeg)) < 1e-10);
    assert(phaseConcentration > 1 - 1e-10 && polarityCorrelation < -0.99);
    assert(strcmp(localPhaseVerdict(meanPhaseDeg, phaseConcentration, ...
        nnz(reliablePhase)), '倾向反相（海面候选）'));
    boundarySegment = localCenteredSegment(oppositeSignal, 0, 29);
    assert(numel(boundarySegment) == 59 && all(isfinite(boundarySegment)));
    tests(3) = "centered_acf_real_cepstrum_and_relative_phase_spectrum";
    localPrintTestProgress(3, 8);

    mainSample = 2880;
    lag = 230;
    assert(mainSample + lag == 3110);
    tests(4) = "lag_to_absolute_sample";
    localPrintTestProgress(4, 8);

    dataset = localBuildDataset(diagnosticsRoot, audioRoot, false);
    assert(height(dataset) == 199, ...
        '当前 recall diagnostics 应恰好映射 199 条有效 WAV，实际为 %d。', height(dataset));
    assert(numel(unique(localNormalizePath(dataset.relative_path))) == height(dataset));
    selectorItems = localSampleSelectorItems(dataset);
    assert(numel(selectorItems) == height(dataset));
    assert(startsWith(selectorItems(1), '001 | ') && ...
        contains(selectorItems(1), dataset.voyage(1)));
    tests(5) = "diagnostic_mapping_and_sample_selector_199";
    localPrintTestProgress(5, 8);

    annotations = localCreateAnnotationTable(dataset(1:2, :));
    annotations.main_sample(1) = 2880;
    annotations.multipath1_sample(1) = 3110;
    annotations.multipath1_quality(1) = "overlapping";
    annotations.status(1) = "completed";
    annotations = localSortAndDeriveRow(annotations, 1, annotations.sample_rate(1));
    [valid, message] = localValidateAnnotationRow(annotations(1, :), ...
        annotations.sample_count(1), true);
    assert(valid, message);
    assert(annotations.multipath1_delay_samples(1) == 230);
    assert(width(annotations) == 22);
    tests(6) = "csv_schema_and_status_validation";
    localPrintTestProgress(6, 8);

    temporaryDirectory = tempname;
    mkdir(temporaryDirectory);
    temporaryCleanup = onCleanup(@() rmdir(temporaryDirectory, 's'));
    temporaryCsv = fullfile(temporaryDirectory, 'annotations.csv');
    annotations.notes(1) = "中文备注：主径清晰";
    localAtomicWriteTable(annotations, temporaryCsv);
    roundTrip = readtable(temporaryCsv, 'TextType', 'string', ...
        'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
    assert(height(roundTrip) == 2 && width(roundTrip) == 22);
    assert(roundTrip.notes(1) == annotations.notes(1));
    assert(roundTrip.multipath1_quality(1) == "overlapping");
    tests(7) = "utf8_csv_round_trip";
    localPrintTestProgress(7, 8);

    annotations.main_sample(2) = 2880;
    annotations.multipath1_sample(2) = 3780;
    annotations.status(2) = "completed";
    annotations = localSortAndDeriveRow(annotations, 2, annotations.sample_rate(2));
    legacy = removevars(annotations, {'multipath1_quality', 'multipath2_quality'});
    legacyCsv = fullfile(temporaryDirectory, 'legacy_annotations.csv');
    localAtomicWriteTable(legacy, legacyCsv);
    migrated = localLoadExistingAnnotations(localCreateAnnotationTable(dataset(1:2, :)), ...
        legacyCsv);
    assert(isequaln(migrated.main_sample, annotations.main_sample));
    assert(isequaln(migrated.multipath1_sample, annotations.multipath1_sample));
    assert(migrated.multipath1_quality(1) == "unlabeled");
    assert(migrated.multipath1_quality(2) == "exclude_late");
    assert(all(migrated.multipath2_quality == "not_applicable"));
    tests(8) = "legacy_csv_position_preserving_quality_migration";
    localPrintTestProgress(8, 8);
    clear temporaryCleanup;
    fprintf('\n');

    report = struct('passed', true, 'test_count', numel(tests), ...
        'tests', tests, 'mapped_audio_count', height(dataset), ...
        'matlab_version', version);
end

function localPrintTestProgress(completed, total)
    filled = round(30 * completed / total);
    fprintf('总体进度: [%s%s] %3d%%  MATLAB GUI 自检\r', ...
        repmat('#', 1, filled), repmat('-', 1, 30 - filled), ...
        round(100 * completed / total));
end

function [cepstrum, quefrencyMs] = localRealCepstrum(signal, sampleRate)
    signal = double(signal(:));
    spectrumMagnitude = abs(fft(signal));
    magnitudeFloor = max(max(spectrumMagnitude) * 1e-12, realmin('double'));
    cepstrum = real(ifft(log(max(spectrumMagnitude, magnitudeFloor))));
    cepstrum = fftshift(cepstrum);
    sampleCount = numel(signal);
    quefrencySamples = (-floor(sampleCount / 2):(ceil(sampleCount / 2) - 1))';
    quefrencyMs = 1000 * quefrencySamples / sampleRate;
end

function items = localSampleSelectorItems(fileInfo)
    indexes = (1:height(fileInfo))';
    items = compose('%03d | %s / %s / %s', indexes, ...
        string(fileInfo.voyage), string(fileInfo.pulse_train), ...
        string(fileInfo.click));
end

function index = localFirstIncompleteIndex(annotations)
    index = find(~localAnnotationCompleteMask(annotations), 1, 'first');
    if isempty(index)
        index = 1;
    end
end

function complete = localAnnotationCompleteMask(annotations)
    complete = annotations.status ~= "unlabeled";
    for multipathIndex = 1:2
        sampleName = sprintf('multipath%d_sample', multipathIndex);
        qualityName = sprintf('multipath%d_quality', multipathIndex);
        hasPath = isfinite(annotations.(sampleName));
        qualityComplete = annotations.(qualityName) ~= "unlabeled" & ...
            annotations.(qualityName) ~= "not_applicable";
        complete = complete & (~hasPath | qualityComplete);
    end
end

function summary = localAnnotationSummary(row, sampleRate)
    summary = sprintf('主径：%s\n多径1：%s [%s]\n多径2：%s [%s]', ...
        localFormatSample(row.main_sample, sampleRate, NaN), ...
        localFormatSample(row.multipath1_sample, sampleRate, row.main_sample), ...
        localQualityDisplayName(row.multipath1_quality), ...
        localFormatSample(row.multipath2_sample, sampleRate, row.main_sample), ...
        localQualityDisplayName(row.multipath2_quality));
end

function name = localQualityDisplayName(quality)
    switch string(quality)
        case "not_applicable"
            name = '不适用';
        case "unlabeled"
            name = '未补充';
        case "clean"
            name = '清晰多径';
        case "overlapping"
            name = '混叠多径';
        case "reverberant"
            name = '混响尾部';
        case "uncertain"
            name = '不确定';
        case "exclude_late"
            name = '1.5 ms后排除';
        otherwise
            name = char(string(quality));
    end
end

function text = localFormatSample(sample, sampleRate, mainSample)
    if ~isfinite(sample)
        text = '未标注';
        return;
    end
    if isfinite(mainSample)
        text = sprintf('%d（%.6f ms，延迟 %.6f ms）', round(sample), ...
            1000 * sample / sampleRate, 1000 * (sample - mainSample) / sampleRate);
    else
        text = sprintf('%d（%.6f ms）', round(sample), 1000 * sample / sampleRate);
    end
end

function name = localTargetDisplayName(target)
    switch string(target)
        case "main"
            name = '主径';
        case "multipath1"
            name = '多径1';
        case "multipath2"
            name = '多径2';
        otherwise
            name = char(string(target));
    end
end

function key = localNormalizePath(pathValue)
    key = lower(replace(string(pathValue), '\', '/'));
end

function index = localLagIndex(lags, lag)
    index = round(lag - lags(1)) + 1;
    if index < 1 || index > numel(lags) || lags(index) ~= lag
        index = NaN;
    end
end

function index = localNearestIndex(values, target)
    [~, index] = min(abs(values - target));
end

function values = localNumericColumn(column)
    if isnumeric(column)
        values = double(column);
    else
        values = str2double(string(column));
    end
end

function value = localScalarDouble(value)
    if isnumeric(value)
        value = double(value(1));
    else
        value = str2double(string(value(1)));
    end
end

function equal = localRowsEqual(left, right)
    left = left(:, right.Properties.VariableNames);
    equal = isequaln(table2cell(left), table2cell(right));
end

function value = localOnOff(condition)
    if condition
        value = 'on';
    else
        value = 'off';
    end
end

function localDeleteIfExists(path)
    if isfile(path)
        delete(path);
    end
end

function valid = localIsTextScalar(value)
    valid = (ischar(value) && isrow(value)) || (isstring(value) && isscalar(value));
end
