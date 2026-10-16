# Unattended PPO training on the real game, with checkpoint resume and GD
# crash recovery.
#
#   powershell -ExecutionPolicy Bypass -File .\overnight.ps1 [-Level main:1]
#
# What it does, in a loop (up to $MaxRestarts failures):
#   1. gdctl prepare-training --level $Level: launches GD if needed (or
#      restarts it if hung), disables Click Between Frames + QOLMod, opens the
#      level in normal mode, turns on prevent_pause, probes the bridge.
#   2. Trains (resuming from the newest checkpoint under runs\$RunFilter).
#   3. If training fails: if GD is dead or unresponsive, gdctl restart; then
#      wait $RetryDelay s and go to 1 (prepare-training --reopen also covers
#      "not in the level").
#
# It holds the gdctl game lock as $Owner the whole time (refreshed every
# 30 min) and releases it on exit.
#
# Stop it:   python -m gdctl stop-training      (or create logs\overnight.stop)
#   The loop notices within ~15 s, stops the trainer (the last checkpoint is at
#   most checkpoint_every updates old), releases the lock and exits.
# Logs: logs\gd_overnight.log (this script), logs\gd_overnight-train.log
# (trainer output, appended per attempt).

param(
    [string]$Level = "main:1",
    [string]$Config = "configs/gd_overnight.yaml",
    [string]$RunFilter = "*gd-overnight*",   # run folders searched for checkpoints
    [string]$Owner = "overnight-training",
    [int]$MaxRestarts = 200,
    [int]$RetryDelay = 30,
    [int]$StartDelay = 0
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = (Resolve-Path (Join-Path $PSScriptRoot "..\geometry-dash-rl\.venv\Scripts\python.exe")).Path
# GPU (agent.device in the config). CUDA_MODULE_LOADING=LAZY is what made the
# CUDA context start again on this laptop (2026-10-11); a config with
# device: cpu still works. To hide the GPU: $env:CUDA_VISIBLE_DEVICES = "-1".
$env:CUDA_MODULE_LOADING = "LAZY"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:GDCTL_OWNER = $Owner
New-Item -ItemType Directory -Force logs | Out-Null
$log = Join-Path $PSScriptRoot "logs\gd_overnight.log"
$trainLog = Join-Path $PSScriptRoot "logs\gd_overnight-train.log"
$trainErr = Join-Path $PSScriptRoot "logs\gd_overnight-train.err"
$stopFile = Join-Path $PSScriptRoot "logs\overnight.stop"
$pidFile = Join-Path $PSScriptRoot "logs\overnight.pid"

function Say([string]$msg) {
    $line = "=== $(Get-Date -Format s) $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Gdctl([string[]]$cmdArgs) {
    # Returns the parsed JSON reply (or $null) and logs a one-line summary.
    $out = & $py -m gdctl @cmdArgs 2>&1 | Out-String
    $obj = $null
    try { $obj = $out | ConvertFrom-Json } catch { }
    $ok = if ($obj) { $obj.ok } else { $false }
    $short = ($out -replace "\s+", " ")
    if ($short.Length -gt 400) { $short = $short.Substring(0, 400) + "..." }
    Say "gdctl $($cmdArgs -join ' ') -> ok=$ok $short"
    return $obj
}

function GameHealthy {
    $st = Gdctl @("status")
    return ($st -and $st.running -and $st.responsive)
}

Remove-Item $stopFile -ErrorAction SilentlyContinue
# Keep Windows from sleeping while this runs (it did on 2026-10-12 at 22:15
# and training stood still for 22 h). Released when the script exits.
Add-Type -Namespace GdRl -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[GdRl.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
Set-Content -Path $pidFile -Value $PID
if ($StartDelay -gt 0) { Start-Sleep $StartDelay }

$lock = Gdctl @("lock", "acquire", "--note", "overnight.ps1 level=$Level pid=$PID", "--wait", "60")
if (-not ($lock -and $lock.ok)) {
    Say "could not get the game lock; another agent is using the game. Exiting."
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    exit 3
}

$failures = 0
$finished = $false
$stopped = $false
try {
    while ($failures -le $MaxRestarts -and -not $finished -and -not $stopped) {
        # 1. Game up, level open, bridge ready.
        # After a failure, reopen the level so the attempt starts clean.
        # A downloaded online level (saved:<id>) loses its level data when GD
        # restarts, so fetch it again first (cheap if it is still there).
        if ($Level -like "saved:*") {
            Gdctl @("download-level", $Level.Substring(6), "--no-song") | Out-Null
        }
        $prepArgs = @("prepare-training", "--level", $Level)
        if ($failures -gt 0) { $prepArgs += "--reopen" }
        $prep = Gdctl $prepArgs
        if (-not ($prep -and $prep.ok)) {
            Say "prepare-training failed; restarting the game"
            Gdctl @("restart") | Out-Null
            $prep = Gdctl @("prepare-training", "--level", $Level, "--reopen")
        }
        if (-not ($prep -and $prep.ok)) {
            $failures++
            Say "game not ready (failure $failures/$MaxRestarts); retrying in $RetryDelay s"
            Start-Sleep $RetryDelay
            continue
        }

        # 2. Train, resuming from the newest checkpoint.
        $trainArgs = @("-m", "gdrl.train", "--config", $Config)
        $latest = Get-ChildItem "runs\$RunFilter\checkpoints\*.pt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime | Select-Object -Last 1
        if ($latest) {
            $trainArgs += @("--set", "train.resume=$($latest.FullName -replace '\\', '/')")
        }
        Say "start training (failures so far $failures): $($trainArgs -join ' ')"
        Add-Content -Path $trainLog -Value "=== $(Get-Date -Format s) $($trainArgs -join ' ')"
        $proc = Start-Process -FilePath $py -ArgumentList $trainArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput "$trainLog.part" -RedirectStandardError $trainErr
        $null = $proc.Handle   # PS 5.1: keep the handle so ExitCode is available
        $lastRefresh = Get-Date
        while (-not $proc.HasExited) {
            Start-Sleep 15
            if (Test-Path $stopFile) {
                Say "stop requested; stopping the trainer"
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                $stopped = $true
                break
            }
            # A minimized GD stops running frames and the trainer times out;
            # restore it behind other windows (no focus stealing).
            & $py -m gdctl unminimize *> $null
            if (((Get-Date) - $lastRefresh).TotalMinutes -ge 30) {
                Gdctl @("lock", "acquire") | Out-Null   # same owner = refresh
                $lastRefresh = Get-Date
            }
        }
        $proc.WaitForExit()
        $code = $proc.ExitCode
        if (Test-Path "$trainLog.part") {
            Get-Content "$trainLog.part" | Add-Content -Path $trainLog
            Remove-Item "$trainLog.part" -ErrorAction SilentlyContinue
        }
        if ($stopped) { break }
        if ($code -eq 0) {
            Say "training finished"
            $finished = $true
            break
        }

        # 3. Failure: recover the game if it is the problem.
        $failures++
        Say "trainer exited with code $code (failure $failures/$MaxRestarts); see $trainErr"
        if (-not (GameHealthy)) {
            Say "game dead or unresponsive; restarting it"
            Gdctl @("restart") | Out-Null
        }
        Start-Sleep $RetryDelay
    }
    if ($failures -gt $MaxRestarts) { Say "too many failures; giving up (checkpoints are kept)" }
}
finally {
    Gdctl @("lock", "release") | Out-Null
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    Remove-Item $stopFile -ErrorAction SilentlyContinue
    Say "overnight.ps1 exiting"
}
