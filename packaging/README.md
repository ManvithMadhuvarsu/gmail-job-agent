# Running MailAI as a background service

These templates run the MailAI daemon at startup on each OS so the buyer does
not have to keep a terminal open.

All three assume:

- MailAI is installed at `~/mailai` (or `%USERPROFILE%\mailai` on Windows).
- A Python virtual env exists at `~/mailai/.venv` with `pip install -r requirements.txt` already run.
- `~/mailai/.env` is filled in with Gmail/LLM/license values.
- Initial OAuth has been completed (`python cli.py run` once, or `/login` on the web).

## Linux (systemd user unit)

```bash
bash packaging/linux/install.sh
```

Status / logs:

```bash
systemctl --user status mailai
journalctl --user -u mailai -f
```

## macOS (launchd LaunchAgent)

```bash
bash packaging/macos/install.sh
```

Status / logs:

```bash
launchctl list | grep com.mailai.daemon
tail -f ~/mailai/data/daemon.log
```

## Windows (Scheduled Task)

In a PowerShell prompt opened *inside the repo*:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\install-service.ps1
```

Status / logs:

```powershell
Get-ScheduledTask -TaskName MailAI
Get-Content data\agent.log -Tail 40 -Wait
```

Remove:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\uninstall-service.ps1
```

## Sanity check

After install, hit `/health` (when running the web surface) or run:

```bash
python cli.py health
```

You should see a recent `last_run_at` and `last_heartbeat_at`.
