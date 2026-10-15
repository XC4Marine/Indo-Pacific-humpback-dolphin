[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'
$ExperimentRoot = Split-Path -Parent $PSCommandPath
$ToolboxRoot = Join-Path $ExperimentRoot 'third_party\Acoustics-Toolbox'
$Repository = 'https://github.com/oalib-acoustics/Acoustics-Toolbox.git'
$BuildFlags = '-O2 -std=gnu -ffast-math -funroll-loops -fomit-frame-pointer -I../misc -I../tslib'

function Write-Stage([int]$Percent, [string]$Status) {
    Write-Progress -Activity 'Bellhop v2 安装与自检' -Status $Status -PercentComplete $Percent
}

Write-Stage 2 '检查编译器和官方源码'
foreach ($tool in @('git', 'gfortran', 'mingw32-make', 'python')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "找不到 $tool。请安装后重新运行此脚本。"
    }
}

if (-not (Test-Path -LiteralPath $ToolboxRoot)) {
    Write-Stage 10 '下载官方 Acoustics Toolbox 源码'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ToolboxRoot) | Out-Null
    & git clone --depth 1 $Repository $ToolboxRoot
    if ($LASTEXITCODE -ne 0) { throw '官方 Acoustics Toolbox 下载失败。' }
} elseif ($Force) {
    Write-Stage 10 '复用已下载的官方源码并强制重新编译'
}

Write-Stage 25 '记录工具链和源码版本'
$commit = (& git -C $ToolboxRoot rev-parse HEAD).Trim()
$versions = [ordered]@{
    source_url       = $Repository
    git_commit       = $commit
    license          = 'GPL-3.0-or-later (see third_party/Acoustics-Toolbox/LICENSE)'
    gfortran         = ((& gfortran --version | Select-Object -First 1) -join '')
    mingw32_make     = ((& mingw32-make --version | Select-Object -First 1) -join '')
    python           = ((& python --version) -join '')
    built_utc        = [DateTime]::UtcNow.ToString('o')
}
$versions | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ExperimentRoot 'toolchain_versions.json') -Encoding utf8

Write-Stage 42 '编译 Bellhop 必需的 misc 库'
& mingw32-make -C (Join-Path $ToolboxRoot 'misc') "FC=gfortran" "FFLAGS=$BuildFlags" all
if ($LASTEXITCODE -ne 0) { throw 'misc 库编译失败。' }

Write-Stage 68 '编译 Bellhop'
& mingw32-make -C (Join-Path $ToolboxRoot 'Bellhop') "FC=gfortran" "FFLAGS=$BuildFlags" bellhop.exe
if ($LASTEXITCODE -ne 0) { throw 'Bellhop 编译失败。' }
$BellhopExe = Join-Path $ToolboxRoot 'Bellhop\bellhop.exe'
if (-not (Test-Path -LiteralPath $BellhopExe)) { throw '未生成 bellhop.exe。' }

Write-Stage 82 '运行官方风格的浅海 arrivals 自检'
$SelfTest = Join-Path $ExperimentRoot 'self_test'
New-Item -ItemType Directory -Force -Path $SelfTest | Out-Null
$Env = @"
'v2 shallow-water arrivals self-test'
100000.0
1
'PVWT'
241 0.0 24.0
0.0 1526.75 /
24.0 1527.15 /
'A' 0.0
24.0 1700.0 0.8 1.8 0.0 0.0 /
1
10.0 /
1
20.0 /
1
0.100 /
'A'
4001
-89.0 89.0 /
0.10 24.0 0.20
"@
$EnvPath = Join-Path $SelfTest 'shallow_arrivals.env'
Set-Content -LiteralPath $EnvPath -Value $Env -Encoding ascii
Push-Location $SelfTest
try {
    & $BellhopExe 'shallow_arrivals' *> 'shallow_arrivals.log'
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $SelfTest 'shallow_arrivals.arr'))) {
        throw 'Bellhop 自检未生成 ASCII arrivals 文件。请检查 self_test/shallow_arrivals.log。'
    }
} finally { Pop-Location }

Write-Stage 94 '用 Python 解析器验证 arrivals'
& python (Join-Path $ExperimentRoot 'run_experiment_v2.py') --self-test
if ($LASTEXITCODE -ne 0) { throw 'Python 未能解析 Bellhop ASCII arrivals 文件。' }

Write-Stage 100 '完成'
Write-Host "Bellhop 已就绪：$BellhopExe"
Write-Host "官方源码提交：$commit"
