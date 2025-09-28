# Restarting Celery Worker on Windows

## Quick Commands

### Find and Kill Existing Celery Workers
```bash
# PowerShell method (recommended)
powershell -Command "Get-Process python* | Where-Object {$_.CommandLine -like '*celery*worker*'} | Stop-Process -Force"

# Alternative: taskkill with escaped slashes (for Git Bash)
taskkill //F //IM python.exe //FI "WINDOWTITLE eq *celery*"
```

### Start New Celery Worker with .venv312
```bash
# Run in background with correct virtual environment
powershell -Command "& '.\.venv312\Scripts\python.exe' -m celery -A app.celery_app worker --loglevel=info --pool=solo"
```

## Step-by-Step Process

1. **Check for existing Celery processes:**
   ```bash
   powershell -Command "Get-Process python* | Where-Object {$_.CommandLine -like '*celery*worker*'} | Select-Object Id, ProcessName, CommandLine"
   ```

2. **Kill old workers** (if any found)

3. **Start fresh worker** with `.venv312` environment

## Important Notes
- Virtual environment is `.venv312` (with dot prefix), not `venv312`
- Use `--pool=solo` flag for Windows compatibility
- When using Git Bash, escape forward slashes with `//` for Windows commands
- PowerShell commands don't need slash escaping (they use `-` for parameters)

## Common Issues
- **"Already running" message during startup**: Old worker from wrong venv - kill and restart
- **Frontend shows "no celery worker"**: Worker using wrong virtual environment
- **Path not found**: Remember the dot prefix in `.venv312`