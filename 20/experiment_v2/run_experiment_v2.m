function run_experiment_v2(varargin)
%RUN_EXPERIMENT_V2 Reproducible 20--190 kHz Bellhop shallow-water experiment.
%   RUN_EXPERIMENT_V2('Resume',true,'Force',false,'Workers',0) processes the
%   SNR >= 90 dB clicks. Workers is accepted to make invocations stable across
%   serial/parallel settings; Bellhop jobs are deliberately launched serially so
%   that the external solver log and the checkpoint update are atomic.

p = inputParser;
addParameter(p, 'Resume', true, @(x)islogical(x) && isscalar(x));
addParameter(p, 'Force', false, @(x)islogical(x) && isscalar(x));
addParameter(p, 'Workers', 0, @(x)isnumeric(x) && isscalar(x) && x >= 0 && mod(x,1)==0);
parse(p, varargin{:});
opt = p.Results;

root = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(root));
cfg = make_config(root, repoRoot, opt);
ensure_runtime(cfg);
cleanupPath = addpath_temporarily(fullfile(cfg.toolboxRoot, 'Matlab', 'ReadWrite')); %#ok<NASGU>

ssp = prepare_ssp(cfg);
manifest = prepare_manifest(cfg, ssp);
write_table_atomic(manifest, cfg.manifestCsv);
fprintf('第20实验 v2：%d 条 click，%d 个频点，共 %d 个工作单元。\n', ...
    height(manifest), numel(cfg.freqHz), height(manifest) * numel(cfg.freqHz));
if opt.Workers > 0
    fprintf('Workers=%d 已记录；为确保外部 Bellhop 日志和断点原子性，本版本按稳定清单顺序提交任务。\n', opt.Workers);
end

started = tic;
doneUnits = 0; okClicks = 0; failedClicks = 0;
for k = 1:height(manifest)
    if opt.Resume && ~opt.Force && is_completed(manifest(k,:), cfg)
        doneUnits = doneUnits + numel(cfg.freqHz);
        okClicks = okClicks + (manifest.status(k) == "success");
        failedClicks = failedClicks + (manifest.status(k) == "failed");
        print_progress(doneUnits, cfg, k, 0, okClicks, failedClicks, started);
        continue
    end
    try
        result = process_click(manifest(k,:), k, ssp, cfg, started, doneUnits, okClicks, failedClicks);
        manifest.status(k) = string(result.status);
        manifest.successful_frequency_count(k) = result.successfulFrequencyCount;
        manifest.first_arrival_s(k) = result.firstArrival;
        manifest.arrivals_10ms(k) = result.arrivals10ms;
        manifest.rms_delay_spread_s(k) = result.rmsDelaySpread;
        manifest.max_excess_delay_s(k) = result.maxExcessDelay;
        manifest.raw_rms_gain(k) = result.rawRmsGain;
        manifest.peak_gain(k) = result.peakGain;
        manifest.preview_scale(k) = result.previewScale;
        manifest.runtime_s(k) = result.runtime;
        manifest.error_message(k) = string(result.errorMessage);
        if result.status == "success", okClicks = okClicks + 1; else, failedClicks = failedClicks + 1; end
    catch ME
        manifest.status(k) = "failed";
        manifest.error_message(k) = string(getReport(ME, 'basic', 'hyperlinks', 'off'));
        manifest.runtime_s(k) = NaN;
        failedClicks = failedClicks + 1;
        write_failure_log(manifest(k,:), cfg, ME);
    end
    doneUnits = doneUnits + numel(cfg.freqHz);
    write_table_atomic(manifest, cfg.manifestCsv); % click-level checkpoint
    write_table_atomic(manifest, cfg.summaryCsv);
    print_progress(doneUnits, cfg, k, numel(cfg.freqHz), okClicks, failedClicks, started);
end
generate_report(manifest, ssp, cfg);
fprintf('完成：成功 %d，失败 %d。报告：%s\n', okClicks, failedClicks, cfg.reportHtml);
end

function cfg = make_config(root, repoRoot, opt)
cfg = struct;
cfg.root = root; cfg.repoRoot = repoRoot;
cfg.dataRoot = fullfile(repoRoot, '00_Data');
cfg.inputCsv = fullfile(cfg.dataRoot, 'SNR_ge_60dB_click_paths.csv');
cfg.sspCsv = fullfile(cfg.dataRoot, '浅海信道24m.csv');
cfg.toolboxRoot = fullfile(root, 'third_party', 'Acoustics-Toolbox');
cfg.bellhopExe = fullfile(cfg.toolboxRoot, 'Bellhop', 'bellhop.exe');
cfg.outputRoot = fullfile(root, 'results');
cfg.clickRoot = fullfile(cfg.outputRoot, 'clicks');
cfg.figureRoot = fullfile(cfg.outputRoot, 'figures');
cfg.manifestCsv = fullfile(cfg.outputRoot, 'click_manifest.csv');
cfg.summaryCsv = fullfile(cfg.outputRoot, 'result_summary.csv');
cfg.reportHtml = fullfile(cfg.outputRoot, 'report_zh.html');
cfg.configJson = fullfile(cfg.outputRoot, 'config.json');
cfg.cleanSspCsv = fullfile(cfg.outputRoot, 'clean_ssp.csv');
cfg.seed = 20260907; cfg.freqHz = (20:5:190)' * 1e3;
cfg.waterDepthM = 24; cfg.receiverDepthM = 20;
cfg.surface = 'flat pressure-release';
cfg.bottom = struct('compressional_speed_mps',1700,'density_gcc',1.8,'attenuation_db_per_lambda',0.8);
cfg.rayCount = 4001; cfg.retryRayCount = 8001; cfg.rayAngles = [-89,89]; cfg.retryRayAngles = [-89.9,89.9];
cfg.arrivalWindowS = .010; cfg.relativeThreshold = 10^(-80/20);
cfg.resume = opt.Resume; cfg.force = opt.Force; cfg.workers = opt.Workers;
cfg.hash = string(hash('SHA-256', jsonencode(rmfield(cfg, {'hash','root','repoRoot','dataRoot','toolboxRoot','outputRoot','clickRoot','figureRoot','manifestCsv','summaryCsv','reportHtml','configJson','cleanSspCsv','inputCsv','sspCsv','bellhopExe','resume','force','workers'}))));
end

function ensure_runtime(cfg)
if ~isfile(cfg.inputCsv), error('找不到输入清单：%s', cfg.inputCsv); end
if ~isfile(cfg.sspCsv), error('找不到声速剖面：%s', cfg.sspCsv); end
if ~isfile(cfg.bellhopExe)
    error('找不到 Bellhop：%s。请先在 %s 运行 setup_bellhop.ps1。', cfg.bellhopExe, cfg.root);
end
mkdir_if_needed(cfg.outputRoot); mkdir_if_needed(cfg.clickRoot); mkdir_if_needed(cfg.figureRoot);
if isfile(cfg.configJson)
    old = jsondecode(fileread(cfg.configJson));
    if ~isfield(old,'hash') || ~strcmp(string(old.hash), cfg.hash)
        error('配置哈希不一致，拒绝续跑。请使用 Force=true 并移走旧 results，或恢复原配置。');
    end
else
    write_text_atomic(cfg.configJson, jsonencode(struct('hash',cfg.hash,'config',cfg), 'PrettyPrint', true));
end
end

function ssp = prepare_ssp(cfg)
raw = read_ssp_csv(cfg.sspCsv);
imax = find(raw.pressure_dbar == max(raw.pressure_dbar), 1, 'first');
raw = raw(1:imax,:); % only downcast
localMedian = movmedian(raw.sound_speed_mps, 5, 'Endpoints', 'shrink');
raw.outlier = abs(raw.sound_speed_mps - localMedian) > 2;
removedOutliers = sum(raw.outlier);
raw = raw(~raw.outlier,:);
raw.depth_m = pressure_to_depth_0deg(raw.pressure_dbar);
[z,order] = sort(raw.depth_m); c = raw.sound_speed_mps(order);
[z,ia] = unique(z, 'stable'); c = c(ia);
surface = median(raw.sound_speed_mps(raw.pressure_dbar >= 0 & raw.pressure_dbar <= .5), 'omitnan');
if ~isfinite(surface), error('0--0.5 dBar 内没有可用表层声速。'); end
z = [0; z; cfg.waterDepthM]; c = [surface; c; c(end)];
[z,ia] = unique(z, 'stable'); c = c(ia);
if any(diff(z)<=0) || z(1) ~= 0 || z(end) ~= cfg.waterDepthM, error('清洗后的 SSP 未严格覆盖 0--24 m。'); end
query = unique([z; (0:.1:cfg.waterDepthM)']);
cp = pchip(z,c,query);
if any(~isfinite(cp)) || any(cp < 1400 | cp > 1600), error('PCHIP SSP 结果不在合理声速范围。'); end
ssp = struct('depth_m',query,'sound_speed_mps',cp,'surface_speed_mps',surface, ...
    'downcast_points_before',imax,'points_after_cleaning',numel(z),'removed_outliers',removedOutliers);
write_table_atomic(table(ssp.depth_m,ssp.sound_speed_mps,'VariableNames',{'depth_m','sound_speed_mps'}), cfg.cleanSspCsv);
end

function raw = read_ssp_csv(path)
lines = readlines(path, 'Encoding','UTF-8'); idx = find(strcmp(strtrim(lines),'[data]'),1,'first');
if isempty(idx), error('SSP 文件不含 [data] 区段。'); end
t = readtable(path, 'NumHeaderLines',idx, 'VariableNamingRule','preserve');
raw = table(double(t.('Pressure (dBar)')), double(t.('SV (m/s)')), 'VariableNames',{'pressure_dbar','sound_speed_mps'});
raw = raw(isfinite(raw.pressure_dbar) & isfinite(raw.sound_speed_mps),:);
end

function z = pressure_to_depth_0deg(p)
g0 = 9.780318; z = (((-1.82e-15.*p + 2.279e-10).*p - 2.2512e-5).*p + 9.72659).*p ./ g0;
end

function manifest = prepare_manifest(cfg, ssp)
input = readtable(cfg.inputCsv, 'VariableNamingRule','preserve');
selected = input(double(input.SNR_dB) >= 90,:);
selected = sortrows(selected, {'TrainNumber','PulseNumber'});
if height(selected) ~= 110, error('SNR >= 90 dB 应为 110 条，实际为 %d 条。', height(selected)); end
if numel(unique(selected.TrainNumber)) ~= 69, error('PulseTrain 数量应为 69。'); end
if any(~selected.PulseWavExists) || any(~isfile(string(selected.PulseWavPath))), error('存在缺失输入 WAV。'); end
validate_wavs(selected.PulseWavPath);
rng(cfg.seed, 'twister'); n = height(selected);
sourceDepth = 1 + 21*rand(n,1); rangeM = 50 + 150*rand(n,1);
clickId = compose('click_%03d_train_%03d_pulse_%03d', (1:n)', selected.TrainNumber, selected.PulseNumber);
out = fullfile(cfg.clickRoot, cellstr(clickId));
manifest = table(clickId, selected.TrainNumber, selected.PulseNumber, selected.SNR_dB, string(selected.PulseWavPath), ...
    sourceDepth, repmat(cfg.receiverDepthM,n,1), rangeM, repmat(cfg.seed,n,1), out, ...
    repmat("pending",n,1), zeros(n,1), nan(n,1), zeros(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), nan(n,1), repmat("",n,1), ...
    'VariableNames', {'click_id','train_number','pulse_number','snr_db','input_wav','source_depth_m','receiver_depth_m','range_m','random_seed','output_path','status','successful_frequency_count','first_arrival_s','arrivals_10ms','rms_delay_spread_s','max_excess_delay_s','raw_rms_gain','peak_gain','preview_scale','runtime_s','error_message'});
if isfile(cfg.manifestCsv) && cfg.resume && ~cfg.force
    old = readtable(cfg.manifestCsv, 'TextType','string');
    if height(old) == height(manifest) && all(old.click_id == manifest.click_id)
        columns = intersect(old.Properties.VariableNames, manifest.Properties.VariableNames);
        manifest(:,columns) = old(:,columns);
    end
end
end

function validate_wavs(paths)
for k = 1:numel(paths)
    a = audioinfo(paths{k});
    if a.SampleRate ~= 576000 || a.NumChannels ~= 1 || a.TotalSamples ~= 116 || a.BitsPerSample ~= 32
        error('WAV 格式不符：%s (Fs=%g, channels=%g, samples=%g, bits=%g)。', paths{k},a.SampleRate,a.NumChannels,a.TotalSamples,a.BitsPerSample);
    end
end
end

function result = process_click(row, clickIndex, ssp, cfg, started, doneBefore, okCount, failCount)
t = tic; clickDir = row.output_path; mkdir_if_needed(clickDir); mkdir_if_needed(fullfile(clickDir,'bellhop'));
[x,fs] = audioread(row.input_wav); x = x(:,1); freqs = cfg.freqHz;
allArrivals = cell(numel(freqs),1); rawFiles = strings(numel(freqs),1); retryUsed = false(numel(freqs),1);
frequencyStatus = table(freqs, repmat("pending",numel(freqs),1), false(numel(freqs),1), strings(numel(freqs),1), strings(numel(freqs),1), ...
    'VariableNames',{'frequency_hz','status','retried','environment_file','arrivals_file','error_message'});
for fi = 1:numel(freqs)
    fDir = fullfile(clickDir,'bellhop',sprintf('f_%03dkHz',round(freqs(fi)/1e3))); mkdir_if_needed(fDir);
    try
        [arr, usedRetry, envFile] = run_frequency(fDir, freqs(fi), row, ssp, cfg);
        allArrivals{fi} = arr; retryUsed(fi) = usedRetry; rawFiles(fi) = string(envFile);
        frequencyStatus.status(fi) = "success"; frequencyStatus.retried(fi) = usedRetry;
        frequencyStatus.environment_file(fi) = string(envFile); frequencyStatus.arrivals_file(fi) = string(fullfile(fDir,sprintf('channel_%03dkHz.arr',round(freqs(fi)/1e3))));
    catch ME
        frequencyStatus.status(fi) = "failed"; frequencyStatus.error_message(fi) = string(getReport(ME,'basic','hyperlinks','off'));
    end
    write_table_atomic(frequencyStatus,fullfile(clickDir,'frequency_status.csv'));
    print_progress(doneBefore+fi, cfg, clickIndex, fi, okCount, failCount, started);
end
if any(frequencyStatus.status ~= "success")
    error('%d 个频点失败；详见 %s。',sum(frequencyStatus.status ~= "success"),fullfile(clickDir,'frequency_status.csv'));
end
firstArrival = min(cellfun(@(a)min(a.delay_s),allArrivals));
trimmed = cellfun(@(a)gate_arrivals(a, firstArrival, cfg), allArrivals, 'UniformOutput',false);
if any(cellfun(@isempty,trimmed)), error('至少一个频点没有满足 10 ms/-80 dB 门限的到达。'); end
[rx, bandSignals] = synthesize_wideband(x, fs, freqs, trimmed, firstArrival);
rawRms = rms(rx); peak = max(abs(rx)); previewScale = 1/max(peak,eps); preview = rx*previewScale;
audiowrite(fullfile(clickDir,'rx_preview.wav'), preview, fs, 'BitsPerSample',32);
rx_raw = rx; inputClick = x; save(fullfile(clickDir,'rx_raw.mat'),'rx_raw','inputClick','fs','firstArrival','-v7.3');
geometry = table(row.source_depth_m,row.receiver_depth_m,row.range_m,row.random_seed,'VariableNames',{'source_depth_m','receiver_depth_m','range_m','random_seed'});
channel = struct('frequencies_hz',freqs,'all_arrivals',{allArrivals},'trimmed_arrivals',{trimmed},'first_arrival_s',firstArrival, ...
    'geometry',geometry,'ssp',ssp,'config_hash',cfg.hash,'retry_used',retryUsed,'raw_files',rawFiles,'subband_signals',{bandSignals});
save(fullfile(clickDir,'channel.mat'),'channel','-v7.3');
result = struct('status',"success",'successfulFrequencyCount',numel(freqs),'firstArrival',firstArrival, ...
    'arrivals10ms',sum(cellfun(@height,trimmed)),'rmsDelaySpread',delay_spread(trimmed,firstArrival), ...
    'maxExcessDelay',max(cellfun(@(a)max(a.delay_s-firstArrival),trimmed)),'rawRmsGain',rawRms/rms(x), ...
    'peakGain',peak/max(abs(x),eps),'previewScale',previewScale,'runtime',toc(t),'errorMessage',"");
end

function [arr, retried, envFile] = run_frequency(dirPath, freqHz, row, ssp, cfg)
base = sprintf('channel_%03dkHz',round(freqHz/1e3)); envFile = fullfile(dirPath,[base,'.env']);
write_env(envFile,freqHz,row,ssp,cfg,cfg.rayCount,cfg.rayAngles,'A');
run_bellhop(dirPath,base,cfg.bellhopExe);
arrFile = fullfile(dirPath,[base,'.arr']); retried = false;
arr = parse_arrivals(arrFile,freqHz);
if isempty(arr)
    retried = true; write_env(envFile,freqHz,row,ssp,cfg,cfg.retryRayCount,cfg.retryRayAngles,'A');
    run_bellhop(dirPath,base,cfg.bellhopExe); arr = parse_arrivals(arrFile,freqHz);
end
if isempty(arr), error('%.0f Hz 在首次及加密射线重试后均无有效到达。',freqHz); end
end

function write_env(path, freqHz, row, ssp, cfg, nbeams, angles, runType)
fid = fopen(path,'w'); assert(fid>0,'无法写入 %s',path); clean = onCleanup(@()fclose(fid));
fprintf(fid,"'Experiment v2 shallow-water channel'\n%.9f\n1\n'PVWT'\n",freqHz);
fprintf(fid,'%d 0.0 %.3f\n',numel(ssp.depth_m),cfg.waterDepthM);
for k=1:numel(ssp.depth_m), fprintf(fid,'%.6f %.6f /\n',ssp.depth_m(k),ssp.sound_speed_mps(k)); end
fprintf(fid,"'A' 0.0\n%.3f %.1f %.3f %.3f 0.0 0.0 /\n",cfg.waterDepthM,cfg.bottom.compressional_speed_mps,cfg.bottom.attenuation_db_per_lambda,cfg.bottom.density_gcc);
fprintf(fid,'1\n%.6f /\n1\n%.6f /\n1\n%.9f /\n',row.source_depth_m,row.receiver_depth_m,row.range_m/1000);
fprintf(fid,"'%s'\n%d\n%.3f %.3f /\n0.10 %.3f %.6f\n",runType,nbeams,angles(1),angles(2),cfg.waterDepthM,max(.2,row.range_m/1000+.01));
end

function run_bellhop(dirPath, base, exe)
logFile = fullfile(dirPath,[base,'.log']);
cmd = sprintf('cd /d "%s" && "%s" "%s" > "%s" 2>&1',dirPath,exe,base,logFile);
[status,out] = system(cmd); %#ok<ASGLU>
if status ~= 0, error('Bellhop 退出码 %d；日志：%s',status,logFile); end
end

function t = parse_arrivals(arrFile, freqHz)
t = table;
if ~isfile(arrFile), return; end
[A,~] = read_arrivals_asc(arrFile);
if A(1).Narr <= 0, return; end
n = double(A(1).Narr); amp = double(A(1).A(1:n)).'; delay = double(A(1).delay(1:n)).';
gain = abs(amp).*exp(2*pi*freqHz*imag(delay));
t = table(repmat(freqHz,n,1),real(delay),imag(delay),amp,gain,double(A(1).SrcDeclAngle(1:n)).',double(A(1).RcvrDeclAngle(1:n)).',double(A(1).NumTopBnc(1:n)).',double(A(1).NumBotBnc(1:n)).', ...
    'VariableNames',{'frequency_hz','delay_s','delay_imag_s','complex_amplitude','effective_gain','source_angle_deg','receiver_angle_deg','top_bounces','bottom_bounces'});
t = sortrows(t,'delay_s');
end

function keep = gate_arrivals(arr, t0, cfg)
threshold = max(arr.effective_gain)*cfg.relativeThreshold;
keep = arr(arr.delay_s >= t0 & arr.delay_s <= t0+cfg.arrivalWindowS & arr.effective_gain >= threshold,:);
end

function [rx, bands] = synthesize_wideband(x, fs, freqHz, arrivals, t0)
n = 2^nextpow2(numel(x)+round(.012*fs)); f = (0:n-1)'/n*fs; signedF = min(f,fs-f);
centres = freqHz(:); weights = zeros(n,numel(centres));
for k=1:numel(centres), weights(:,k) = max(cos(pi/2*(signedF-centres(k))/10000),0); end
normalizer = sqrt(sum(weights.^2,2)); weights = weights./max(normalizer,eps); weights(signedF<20000 | signedF>190000,:) = 0;
X = fft([x;zeros(n-numel(x),1)]); rx = zeros(n,1); bands = cell(numel(centres),1);
for k=1:numel(centres)
    b = real(ifft(X.*weights(:,k))); analytic = analytic_signal(b); contribution = zeros(n,1);
    for a=1:height(arrivals{k})
        d = arrivals{k}.delay_s(a)-t0;
        shifted = ifft(fft(analytic).*exp(-1i*2*pi*((0:n-1)'/n*fs)*d));
        contribution = contribution + real(arrivals{k}.complex_amplitude(a)*shifted);
    end
    rx = rx + contribution; bands{k} = b;
end
rx = rx(1:round(.012*fs));
end

function xa = analytic_signal(x)
n = numel(x); X = fft(x); h = zeros(n,1); h(1)=1;
if mod(n,2)==0, h(n/2+1)=1; h(2:n/2)=2; else, h(2:(n+1)/2)=2; end
xa = ifft(X.*h);
end

function ds = delay_spread(arrivals,t0)
d=[]; w=[]; for k=1:numel(arrivals), d=[d;arrivals{k}.delay_s-t0]; w=[w;arrivals{k}.effective_gain.^2]; end %#ok<AGROW>
mu=sum(w.*d)/sum(w); ds=sqrt(sum(w.*(d-mu).^2)/sum(w));
end

function generate_report(manifest,ssp,cfg)
close all; f=figure('Visible','off'); plot(ssp.sound_speed_mps,ssp.depth_m,'LineWidth',1.5); set(gca,'YDir','reverse'); grid on; xlabel('声速 (m/s)'); ylabel('深度 (m)'); title('清洗后的实测浅海声速剖面'); saveas(f,fullfile(cfg.figureRoot,'ssp.png')); close(f);
f=figure('Visible','off'); scatter(manifest.range_m,manifest.raw_rms_gain,36,manifest.snr_db,'filled'); grid on; xlabel('距离 (m)'); ylabel('原始 RMS 相对增益'); title('相对增益与距离'); colorbar; saveas(f,fullfile(cfg.figureRoot,'gain_distance.png')); close(f);
f=figure('Visible','off'); histogram(manifest.arrivals_10ms(manifest.status=="success")); grid on; xlabel('10 ms 内到达数'); ylabel('click 数'); title('有效到达统计'); saveas(f,fullfile(cfg.figureRoot,'arrival_counts.png')); close(f);
success = manifest(manifest.status=="success",:); reps = choose_representatives(success);
repHtml = "";
for k=1:height(reps)
    repHtml = repHtml + make_representative_plot(reps(k,:),cfg,k);
end
summary = sprintf('<p>总计 %d 条；成功 %d 条；失败 %d 条。频段：20--190 kHz（35 点）。所有增益为相对数字幅值，未校准为 Pa 或 dB re 1 μPa。</p>',height(manifest),sum(manifest.status=="success"),sum(manifest.status=="failed"));
html = "<!doctype html><html><head><meta charset=\"utf-8\"><title>第20实验 v2 报告</title><style>body{font-family:Microsoft YaHei,Arial;margin:28px}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #bbb;padding:4px}img{max-width:760px;margin:10px}</style></head><body>" + ...
    "<h1>第20实验 v2：Bellhop 浅海宽带信道</h1>" + summary + "<h2>实测 SSP</h2><img src=\"figures/ssp.png\">" + ...
    "<h2>随机几何及增益</h2><img src=\"figures/gain_distance.png\"><img src=\"figures/arrival_counts.png\">" + ...
    "<h2>近、中、远距离代表样本</h2>" + repHtml + "<h2>逐 click 结果</h2>" + table_to_html(manifest) + "</body></html>";
write_text_atomic(cfg.reportHtml,html);
end

function reps=choose_representatives(t)
if isempty(t), reps=t; return; end
[~,i1]=min(t.range_m); [~,i3]=max(t.range_m); [~,i2]=min(abs(t.range_m-median(t.range_m))); reps=t([i1 i2 i3],:);
end

function html = make_representative_plot(row,cfg,index)
d=load(fullfile(row.output_path,'rx_raw.mat')); ch=load(fullfile(row.output_path,'channel.mat'));
rayDir=fullfile(row.output_path,'bellhop','representative_ray'); mkdir_if_needed(rayDir); rayBase='representative_ray';
write_env(fullfile(rayDir,[rayBase,'.env']),100e3,row,ch.channel.ssp,cfg,cfg.rayCount,cfg.rayAngles,'R'); run_bellhop(rayDir,rayBase,cfg.bellhopExe);
name=sprintf('representative_%d.png',index); f=figure('Visible','off','Position',[100 100 1100 900]);
subplot(4,1,1); cleanup=addpath_temporarily(fullfile(cfg.toolboxRoot,'Matlab','Plot')); global units jkpsflag; units='m'; jkpsflag=false; plotray(fullfile(rayDir,rayBase)); title(sprintf('%s：100 kHz Bellhop 射线路径',row.click_id)); clear cleanup;
subplot(4,1,2); h=zeros(numel(ch.channel.frequencies_hz),1); for q=1:numel(h), a=ch.channel.trimmed_arrivals{q}; h(q)=sum(a.complex_amplitude.*exp(-1i*2*pi*ch.channel.frequencies_hz(q)*(a.delay_s-ch.channel.first_arrival_s))); end; plot(ch.channel.frequencies_hz/1e3,20*log10(abs(h)+eps),'o-'); grid on; xlabel('频率 (kHz)'); ylabel('相对幅度 (dB)'); title('截断到达的复信道响应');
subplot(4,1,3); plot((0:numel(d.inputClick)-1)/d.fs*1e3,d.inputClick); grid on; title('传播前波形'); xlabel('时间 (ms)');
subplot(4,1,4); plot((0:numel(d.rx_raw)-1)/d.fs*1e3,d.rx_raw); grid on; title(sprintf('传播后波形，距离 %.1f m',row.range_m)); xlabel('相对到达后时间 (ms)'); saveas(f,fullfile(cfg.figureRoot,name)); close(f);
html=sprintf('<h3>%s（距离 %.2f m）</h3><img src="figures/%s">',row.click_id,row.range_m,name);
end

function html=table_to_html(t)
headers=t.Properties.VariableNames; html='<table><tr>'; for k=1:numel(headers), html=[html,'<th>',headers{k},'</th>']; end; html=[html,'</tr>'];
for r=1:height(t), html=[html,'<tr>']; for c=1:width(t), v=t{r,c}; if isstring(v)||ischar(v), s=char(v); elseif isnumeric(v), s=num2str(v); else, s=char(string(v)); end; html=[html,'<td>',escape_html(s),'</td>']; end; html=[html,'</tr>']; end; html=[html,'</table>'];
end

function s=escape_html(s), s=strrep(strrep(strrep(s,'&','&amp;'),'<','&lt;'),'>','&gt;'); end
function tf=is_completed(row,cfg), tf=(row.status=="success" || row.status=="failed") && isfile(fullfile(row.output_path,'channel.mat')) || (row.status=="failed" && isfile(fullfile(row.output_path,'failure.log'))); end
function print_progress(done,cfg,click,fi,ok,fail,started), elapsed=toc(started); remain=elapsed/max(done,1)*(numel(cfg.freqHz)*110-done); fprintf('[%6.2f%%] click %d / 110，频点 %d / 35；成功/失败 %d/%d；耗时 %s，预计剩余 %s\n',100*done/(110*numel(cfg.freqHz)),click,fi,ok,fail,duration(0,0,elapsed,'Format','hh:mm:ss'),duration(0,0,max(0,remain),'Format','hh:mm:ss')); end
function write_failure_log(row,cfg,ME), mkdir_if_needed(row.output_path); write_text_atomic(fullfile(row.output_path,'failure.log'),getReport(ME,'extended','hyperlinks','off')); end
function mkdir_if_needed(p), if ~isfolder(p), mkdir(p); end, end
function write_table_atomic(t,path), tmp=[path,'.tmp.csv']; writetable(t,tmp); movefile(tmp,path,'f'); end
function write_text_atomic(path,text), tmp=[path,'.tmp']; fid=fopen(tmp,'w','n','UTF-8'); fwrite(fid,text,'char'); fclose(fid); movefile(tmp,path,'f'); end
function cleanup=addpath_temporarily(p), addpath(p); cleanup=onCleanup(@()rmpath(p)); end
