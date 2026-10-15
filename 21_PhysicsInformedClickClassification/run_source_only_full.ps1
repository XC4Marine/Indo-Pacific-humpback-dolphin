$ErrorActionPreference = "Stop"
$OutputDir = "D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_source_only_5seeds"
$PythonExe = "D:\Python_env\toothwhale\python.exe"
$Pipeline = "D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\pipeline.py"
$StatusPath = Join-Path $OutputDir "background_status.json"
$StartedUtc = [DateTime]::UtcNow.ToString("o")

function Write-RunStatus {
    param([string]$State, [Nullable[int]]$ExitCode = $null, [string]$Message = "")
    $status = [ordered]@{
        state = $State; pid = $PID; started_utc = $StartedUtc
        updated_utc = [DateTime]::UtcNow.ToString("o"); exit_code = $ExitCode
        message = $Message; output_dir = $OutputDir
    }
    $temporary = "$StatusPath.tmp"
    $status | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-RunStatus -State "running" -Message "Full five-seed, five-fold source-only ablation is running."
try {
    & $PythonExe $Pipeline --mode full --seeds 20260826 20260827 20260828 20260829 20260830 `
        --output-dir $OutputDir --positive-noise-mix --snr-min-db -15 --snr-max-db 15 `
        --mix-seed 20260826 --conditions recon_bce_source_only recon_physics_bce_source_only --skip-report
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-RunStatus -State "completed" -ExitCode $code -Message "Full run completed and artifact validation passed."
    } else {
        Write-RunStatus -State "failed" -ExitCode $code -Message "Pipeline exited with a non-zero code; completed checkpoints were retained."
    }
    exit $code
} catch {
    Write-RunStatus -State "failed" -ExitCode 1 -Message $_.Exception.Message
    throw
}
