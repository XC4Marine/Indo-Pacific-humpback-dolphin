function run_tests_v2
%RUN_TESTS_V2 Lightweight acceptance tests before the 110-click full run.
root = fileparts(fileparts(mfilename('fullpath'))); repo = fileparts(fileparts(root));
data = fullfile(repo,'00_Data'); toolbox = fullfile(root,'third_party','Acoustics-Toolbox');

% Selection boundary, uniqueness, WAV existence, and exact input format.
t = readtable(fullfile(data,'SNR_ge_60dB_click_paths.csv'));
s = sortrows(t(double(t.SNR_dB)>=90,:),{'TrainNumber','PulseNumber'});
assert(height(s)==110 && numel(unique(s.TrainNumber))==69, 'SNR 筛选数量不符合预期。');
assert(all(double(s.SNR_dB)>=90) && all(s.PulseWavExists) && all(isfile(string(s.PulseWavPath))), '筛选结果存在缺失 WAV。');
for k=1:height(s), a=audioinfo(s.PulseWavPath{k}); assert(a.SampleRate==576000 && a.NumChannels==1 && a.BitsPerSample==32 && a.TotalSamples==116, 'WAV 元数据不一致。'); end

% Seeded geometry must not depend on executor/worker count.
rng(20260907,'twister'); g1=[1+21*rand(height(s),1),50+150*rand(height(s),1)];
rng(20260907,'twister'); g2=[1+21*rand(height(s),1),50+150*rand(height(s),1)];
assert(isequal(g1,g2), 'MT19937 几何不可复现。');

% SSP downcast and outlier rule.
lines=readlines(fullfile(data,'浅海信道24m.csv')); i=find(strcmp(strtrim(lines),'[data]'),1); x=readtable(fullfile(data,'浅海信道24m.csv'),'NumHeaderLines',i,'VariableNamingRule','preserve'); p=double(x.('Pressure (dBar)')); c=double(x.('SV (m/s)')); imax=find(p==max(p),1,'first'); p=p(1:imax); c=c(1:imax); keep=abs(c-movmedian(c,5,'Endpoints','shrink'))<=2; assert(any(abs(c-1518.276)<1e-6) && ~all(keep), '1518.276 m/s 异常点未被识别。'); z=sort(pressure_to_depth_0deg(p(keep))); assert(all(diff(unique(z))>0) && max(z)<24, 'SSP 下放段深度不严格递增。');

% Bellhop executable and the official ASCII reader must agree on self-test output.
assert(isfile(fullfile(toolbox,'Bellhop','bellhop.exe')), 'Bellhop 未编译。请先运行 setup_bellhop.ps1。');
addpath(fullfile(toolbox,'Matlab','ReadWrite')); cleanup=onCleanup(@()rmpath(fullfile(toolbox,'Matlab','ReadWrite'))); %#ok<NASGU>
arrFile=fullfile(root,'self_test','shallow_arrivals.arr'); assert(isfile(arrFile), '缺少 self_test arrivals；请运行 setup_bellhop.ps1。'); [arr,~]=read_arrivals_asc(arrFile); assert(arr(1).Narr>0 && all(isfinite(real(arr(1).delay))), 'ASCII arrivals 解析失败。');

% Energy-complementary subband masks and a unit channel identity check.
fs=576000; n=8192; f=(0:n-1)'/n*fs; sf=min(f,fs-f); centres=(20:5:190)'*1e3; w=zeros(n,numel(centres)); for k=1:numel(centres), w(:,k)=max(cos(pi/2*(sf-centres(k))/10000),0); end; w=w./max(sqrt(sum(w.^2,2)),eps); active=sf>=20000 & sf<=190000; assert(max(abs(sum(w(active,:).^2,2)-1))<1e-12, '子带不是能量互补。');
fprintf('run_tests_v2: 全部预检通过。\n');
end

function z=pressure_to_depth_0deg(p)
z=((( -1.82e-15.*p+2.279e-10).*p-2.2512e-5).*p+9.72659).*p/9.780318;
end
