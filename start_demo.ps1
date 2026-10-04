# start_demo.ps1
# Launches the pipeline and dashboard together, each in its own required
# venv, from a single command — so presenting doesn't mean manually
# activating two environments and typing two commands under pressure.

$root = $PSScriptRoot

Write-Host "Starting main pipeline (venv)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root'; .\venv\Scripts\Activate.ps1; python run_pipeline.py"

Start-Sleep -Seconds 3

Write-Host "Starting proctor dashboard (venv_dashboard)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$root'; .\venv_dashboard\Scripts\Activate.ps1; streamlit run dashboard.py"

Write-Host "`nBoth windows are launching:" -ForegroundColor Green
Write-Host "  - One shows the live camera feed with detection overlays"
Write-Host "  - The other opens the dashboard in your browser automatically"