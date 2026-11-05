$env:PYTHONUTF8 = "1"
Set-Location -Path $PSScriptRoot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Chemotherapy PROMIS Risk Streamlit Dashboard" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
python -m streamlit run app.py
