$ErrorActionPreference = "Stop"
$OutputDir = "D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_source_only_5seeds"
$Runner = "D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\run_source_only_full.ps1"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$stdout = Join-Path $OutputDir "background_stdout.log"
$stderr = Join-Path $OutputDir "background_stderr.log"
$process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner) `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$launch = [ordered]@{
    pid = $process.Id; launched_utc = [DateTime]::UtcNow.ToString("o"); runner = $Runner
    stdout = $stdout; stderr = $stderr; output_dir = $OutputDir
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "background_launch.json") -Encoding UTF8
Start-Sleep -Milliseconds 500
if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) { throw "后台训练进程启动后立即退出。" }
$process | Select-Object Id, ProcessName, StartTime
