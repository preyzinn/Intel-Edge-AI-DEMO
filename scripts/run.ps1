param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "src\edge_ai_demo\app.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "Virtual environment not found. Run '.\scripts\setup.ps1' first." -ErrorAction Continue
    exit 1
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    Write-Error "Streamlit entry point not found: $EntryPoint" -ErrorAction Continue
    exit 1
}

Set-Location $ProjectRoot
Write-Host "Starting Intel Edge AI Demo at http://127.0.0.1:$Port"
& $VenvPython -m streamlit run $EntryPoint `
    --server.address 127.0.0.1 `
    --server.port $Port `
    --server.headless true
exit $LASTEXITCODE
