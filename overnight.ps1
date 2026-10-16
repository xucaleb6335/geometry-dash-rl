# Overnight PPO training on the real game, with automatic resume.
#
#   1. Start GD, disable Click Between Frames and QOLMod, open the level
#      (normal mode, not practice).
#   2. In PowerShell:  cd $HOME\Desktop\geometry-dash-rl-autopilot; .\overnight.ps1
#   3. Within 10 seconds, click into GD and resume it if it paused.
#
# If training stops with an error (connection lost, game hiccup), this waits
# a minute and resumes from the newest checkpoint, up to $MaxRestarts times.
# Output goes to the console and to logs\gd_overnight.log.

param(
    [string]$Config = "configs/gd_overnight.yaml",
    [int]$MaxRestarts = 20,
    [int]$StartDelay = 10
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "..\geometry-dash-rl\.venv\Scripts\python.exe"
# CPU only: the networks are tiny, and the GPU can fail to start a CUDA context.
$env:CUDA_VISIBLE_DEVICES = "-1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\gd_overnight.log"

Write-Host "Starting in $StartDelay s: click into Geometry Dash and resume the level now."
Start-Sleep $StartDelay

for ($restart = 0; $restart -le $MaxRestarts; $restart++) {
    $trainArgs = @("-m", "gdrl.train", "--config", $Config)
    $latest = Get-ChildItem "runs\*gd-overnight*\checkpoints\*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($latest) {
        $trainArgs += @("--set", "train.resume=$($latest.FullName -replace '\\', '/')")
    }
    "=== $(Get-Date -Format s) start (restart $restart) $($trainArgs -join ' ')" | Tee-Object -Append $log
    & $py @trainArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append $log
    if ($LASTEXITCODE -eq 0) {
        "=== $(Get-Date -Format s) finished" | Tee-Object -Append $log
        break
    }
    "=== $(Get-Date -Format s) exited with code $LASTEXITCODE; retrying in 60 s" | Tee-Object -Append $log
    Start-Sleep 60
}
