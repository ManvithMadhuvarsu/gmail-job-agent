# Remove the MailAI Scheduled Task installed by install-service.ps1.
$ErrorActionPreference = "Stop"

$TaskName = "MailAI"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed Scheduled Task '$TaskName'."
} else {
    Write-Host "Scheduled Task '$TaskName' was not present."
}
