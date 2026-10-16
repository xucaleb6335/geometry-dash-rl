# Keeps real-game training running while no agent is using the game.
#
# Run by a Windows scheduled task every 15 minutes (GDRL-Watchdog). If no
# gdrl.train process is running AND the game lock is free (or has not been
# refreshed for $StaleMinutes), it starts overnight.ps1 detached with the job
# in configs/watchdog.json. Agents that want the game take the lock as usual;
# agents that change what should train edit configs/watchdog.json.
#
# Install / remove (once, as the user):
#   schtasks /Create /SC MINUTE /MO 15 /TN GDRL-Watchdog /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File %USERPROFILE%\Desktop\geometry-dash-rl-autopilot\watchdog.ps1"
#   schtasks /Delete /TN GDRL-Watchdog /F

param([int]$StaleMinutes = 60)

Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\watchdog.log"
function Say($msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $log }

if (Test-Path "logs\watchdog.disabled") { exit 0 }

$training = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*gdrl.train*' -or $_.CommandLine -like '*gdrl.search*' }
if ($training) { exit 0 }
$runner = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like '*overnight.ps1*' }
if ($runner) { exit 0 }

if (Test-Path "logs\game.lock") {
    try {
        $lock = Get-Content "logs\game.lock" -Raw | ConvertFrom-Json
        $last = if ($lock.refreshed) { $lock.refreshed } else { $lock.acquired }
        $ageMin = ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$last) / 60
        if ($ageMin -lt $StaleMinutes) { exit 0 }   # an agent is (probably) using the game
        Say "lock held by $($lock.owner) but idle for $([int]$ageMin) min; taking over"
        Remove-Item "logs\game.lock" -Force
    } catch { Say "unreadable lock: $_"; exit 0 }
}

$job = Get-Content "configs\watchdog.json" -Raw | ConvertFrom-Json
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$PSScriptRoot\overnight.ps1`" -Level $($job.level) -Config $($job.config) -RunFilter `"$($job.run_filter)`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = "$PSScriptRoot" }
Say "started training (pid $($r.ProcessId)): $cmd"
