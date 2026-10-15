$ErrorActionPreference = "Stop"
$TaskRoot = "D:\Project_Github\Indo-Pacific-humpback-dolphin\21_v5_ParametricChannel"
$PythonExe = "D:\Python_env\toothwhale\python.exe"
$Output = Join-Path $TaskRoot "outputs_seed20260826"
$Status = Join-Path $Output "background_status.json"
New-Item -ItemType Directory -Path $Output -Force | Out-Null
@{ state="running"; pid=$PID; started_utc=[DateTime]::UtcNow.ToString("o"); output_dir=$Output } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8
try {
  & $PythonExe (Join-Path $TaskRoot "parametric_channel.py") --seed 20260826 --conditions B_parametric_fixed_first C_parametric_unit_energy D_parametric_no_physics --folds 0 1 2 3 4
  @{ state="completed"; pid=$PID; completed_utc=[DateTime]::UtcNow.ToString("o"); exit_code=$LASTEXITCODE; output_dir=$Output } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8
  exit $LASTEXITCODE
} catch {
  @{ state="failed"; pid=$PID; updated_utc=[DateTime]::UtcNow.ToString("o"); message=$_.Exception.Message; output_dir=$Output } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding UTF8
  throw
}
