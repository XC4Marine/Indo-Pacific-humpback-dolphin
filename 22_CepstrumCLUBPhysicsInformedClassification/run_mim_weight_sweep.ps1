$ErrorActionPreference = 'Stop'

$experimentDir = 'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification'
$python = 'D:\Python_env\toothwhale\python.exe'
$script = Join-Path $experimentDir 'experiment.py'
$sweepRoot = Join-Path $experimentDir 'outputs_mim_weight_sweep_positive_mim_v3'
New-Item -ItemType Directory -Path $sweepRoot -Force | Out-Null

$settings = @(
    @{ Weight = '0.03'; Name = '003' },
    @{ Weight = '0.1'; Name = '010' },
    @{ Weight = '0.3'; Name = '030' }
)

Set-Content -LiteralPath (Join-Path $sweepRoot 'overall_progress.log') -Value 'progress: 0/75 models'
$originalOutputDir = Join-Path $experimentDir 'outputs_full_no_cepstrum_positive_mim_5seeds_4conditions'
$restoreArguments = @(
    $script, '--mode', 'full', '--device', 'cuda',
    '--seeds', '20260826', '20260827', '20260828', '20260829', '20260830',
    '--folds', '0', '1', '2', '3', '4',
    '--conditions', 'full', 'without_recon', 'without_mim', 'bce_only',
    '--mim-weight', '1.0', '--max-epochs', '25', '--patience', '5',
    '--bootstrap-runs', '2000', '--batch-size', '16', '--resume',
    '--output-dir', $originalOutputDir
)
$restore = Start-Process -FilePath $python -ArgumentList $restoreArguments -WorkingDirectory $experimentDir `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $sweepRoot 'original_result_recovery_stdout.log') `
    -RedirectStandardError (Join-Path $sweepRoot 'original_result_recovery_stderr.log') -PassThru -Wait
if ($restore.ExitCode -ne 0) {
    throw "original result recovery failed with exit code $($restore.ExitCode)."
}
foreach ($setting in $settings) {
    $outputDir = Join-Path $experimentDir ("outputs_mim_weight_{0}_full_5seeds_5fold" -f $setting.Name)
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $stdout = Join-Path $outputDir 'training_stdout.log'
    $stderr = Join-Path $outputDir 'training_stderr.log'
    $arguments = @(
        $script, '--mode', 'full', '--device', 'cuda',
        '--seeds', '20260826', '20260827', '20260828', '20260829', '20260830',
        '--folds', '0', '1', '2', '3', '4', '--conditions', 'full',
        '--mim-weight', $setting.Weight, '--max-epochs', '25', '--patience', '5',
        '--bootstrap-runs', '2000', '--batch-size', '16', '--resume', '--output-dir', $outputDir
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $experimentDir `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "mim_weight=$($setting.Weight) failed with exit code $($process.ExitCode)."
    }
    $complete = (Get-ChildItem -LiteralPath $experimentDir -Directory -Filter 'outputs_mim_weight_*_full_5seeds_5fold' |
        ForEach-Object { Get-ChildItem -LiteralPath (Join-Path $_.FullName 'runs') -Filter COMPLETE.json -Recurse -ErrorAction SilentlyContinue } |
        Measure-Object).Count
    Set-Content -LiteralPath (Join-Path $sweepRoot 'overall_progress.log') -Value ("progress: {0}/75 models" -f $complete)
}

Set-Content -LiteralPath (Join-Path $sweepRoot 'overall_progress.log') -Value 'progress: 75/75 models; complete'
