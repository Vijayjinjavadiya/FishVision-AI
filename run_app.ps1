$ErrorActionPreference = "Stop"

$python = "C:\Users\jinju\anaconda3\envs\fish\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python for Conda environment 'fish' was not found at: $python"
}

& $python -m streamlit run app.py --server.port 8501
