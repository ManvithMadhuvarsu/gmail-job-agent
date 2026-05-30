# Install MailAI as a per-user Scheduled Task that runs at logon.
# Usage (in an elevated *or* normal PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging/windows/install-service.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python.exe).Source
    Write-Warning "Local .venv not found, using system python: $PythonExe"
}

$TaskName = "MailAI"
$WorkingDir = $RepoRoot
$Arguments = "cli.py daemon"

# Run-only-when-logged-on (no stored password). Restart on failure handled by Scheduler.
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $Arguments `
    -WorkingDirectory $WorkingDir

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartOnFailure `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "MailAI Gmail job-email daemon" | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Host ""
Write-Host "MailAI installed as Scheduled Task '$TaskName'."
Write-Host "  status:  Get-ScheduledTask -TaskName $TaskName"
Write-Host "  stop:    Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  remove:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host "  logs:    Get-Content $WorkingDir\data\agent.log -Tail 40 -Wait"
