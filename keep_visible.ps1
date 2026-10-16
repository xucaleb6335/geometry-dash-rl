# Keeps GD un-minimized while agents use it: a minimized GD stops running
# frames, so the bridge stops answering and clients time out. Restores it
# behind other windows (no focus stealing) every 3 s.
#   powershell -ExecutionPolicy Bypass -File .\keep_visible.ps1
Set-Location $PSScriptRoot
$py = (Resolve-Path "..\geometry-dash-rl\.venv\Scripts\python.exe").Path
while ($true) { & $py -m gdctl unminimize *> $null; Start-Sleep 3 }
