$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$SetupMarker = Join-Path $VenvDirectory ".edge-ai-setup"
$SetupMarkerTemporary = Join-Path $VenvDirectory ".edge-ai-setup.tmp"
$ValidationScriptPath = Join-Path $VenvDirectory ".edge-ai-validate.py"
$ProjectManifest = Join-Path $ProjectRoot "pyproject.toml"
$MinimumPython = [version]"3.10"
$MaximumPythonExclusive = [version]"3.15"

function Get-PythonVersion {
    param(
        [Parameter(Mandatory)] [string]$Executable,
        [string[]]$Arguments = @()
    )

    try {
        $output = & $Executable @Arguments -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return [version]([string]($output | Select-Object -First 1)).Trim()
    }
    catch {
        return $null
    }
}

function Test-SupportedPython {
    param([AllowNull()] [version]$Version)

    return $null -ne $Version -and $Version -ge $MinimumPython -and $Version -lt $MaximumPythonExclusive
}

function Test-64BitPython {
    param(
        [Parameter(Mandatory)] [string]$Executable,
        [string[]]$Arguments = @()
    )

    try {
        & $Executable @Arguments -c "import struct; raise SystemExit(0 if struct.calcsize('P') * 8 == 64 else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-SupportedPython {
    $candidates = @(
        [pscustomobject]@{ Executable = "py"; Arguments = @("-3.12") },
        [pscustomobject]@{ Executable = "py"; Arguments = @("-3.11") },
        [pscustomobject]@{ Executable = "py"; Arguments = @("-3.13") },
        [pscustomobject]@{ Executable = "py"; Arguments = @("-3.14") },
        [pscustomobject]@{ Executable = "py"; Arguments = @("-3.10") },
        [pscustomobject]@{ Executable = "python"; Arguments = @() },
        [pscustomobject]@{ Executable = "python3"; Arguments = @() }
    )

    foreach ($candidate in $candidates) {
        if ($null -eq (Get-Command $candidate.Executable -ErrorAction SilentlyContinue)) {
            continue
        }

        $version = Get-PythonVersion -Executable $candidate.Executable -Arguments $candidate.Arguments
        if ((Test-SupportedPython -Version $version) -and (Test-64BitPython -Executable $candidate.Executable -Arguments $candidate.Arguments)) {
            $displayCommand = (@($candidate.Executable) + $candidate.Arguments) -join " "
            Write-Host "Using $displayCommand (Python $version)."
            return $candidate
        }
    }

    throw "Python 3.10 through 3.14 was not found. Install a supported 64-bit Python from https://www.python.org/downloads/ and try again."
}

function Assert-LastCommandSucceeded {
    param([Parameter(Mandatory)] [string]$Message)

    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

try {
    Set-Location $ProjectRoot
    Write-Host "Project: $ProjectRoot"

    if (Test-Path -LiteralPath $SetupMarker) {
        Remove-Item -LiteralPath $SetupMarker -Force
    }
    if (Test-Path -LiteralPath $SetupMarkerTemporary) {
        Remove-Item -LiteralPath $SetupMarkerTemporary -Force
    }
    if (Test-Path -LiteralPath $ValidationScriptPath) {
        Remove-Item -LiteralPath $ValidationScriptPath -Force
    }

    if (Test-Path -LiteralPath $VenvPython) {
        $venvVersion = Get-PythonVersion -Executable $VenvPython
        if (-not (Test-SupportedPython -Version $venvVersion)) {
            throw "The existing .venv uses unsupported Python $venvVersion. Remove '$VenvDirectory' and run this script again."
        }
        if (-not (Test-64BitPython -Executable $VenvPython)) {
            throw "The existing .venv uses 32-bit Python. Remove '$VenvDirectory', install 64-bit Python, and run this script again."
        }
        Write-Host "Reusing .venv with Python $venvVersion."
    }
    elseif (Test-Path -LiteralPath $VenvDirectory) {
        throw "The existing '$VenvDirectory' is incomplete. Remove it and run this script again."
    }
    else {
        $python = Find-SupportedPython
        $pythonExecutable = $python.Executable
        $pythonArguments = $python.Arguments
        Write-Host "Creating .venv..."
        & $pythonExecutable @pythonArguments -m venv $VenvDirectory
        Assert-LastCommandSucceeded "Could not create the virtual environment."
    }

    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip
    Assert-LastCommandSucceeded "Could not upgrade pip. Check the network connection and proxy settings."

    Write-Host "Installing the benchmark and its required dependencies..."
    & $VenvPython -m pip install --editable $ProjectRoot
    Assert-LastCommandSucceeded "Dependency installation failed. Check the messages above for unavailable wheels, network, disk-space, or proxy errors."

    Write-Host "Checking installed dependency consistency..."
    & $VenvPython -m pip check
    Assert-LastCommandSucceeded "pip found an incompatible dependency set."

    $validationScript = @'
from importlib import metadata

packages = {
    "openvino": "openvino",
    "optimum-intel": "optimum.intel",
    "psutil": "psutil",
    "streamlit": "streamlit",
    "torch": "torch",
    "transformers": "transformers",
}
for distribution in packages:
    print(f"  {distribution} {metadata.version(distribution)}")

import edge_ai_demo  # noqa: F401, E402
import openvino  # noqa: F401, E402
import optimum.intel  # noqa: F401, E402
import psutil  # noqa: F401, E402
import streamlit  # noqa: F401, E402
import torch  # noqa: F401, E402
import transformers  # noqa: F401, E402
print("  edge_ai_demo import OK")
'@
    Write-Host "Validating imports..."
    [IO.File]::WriteAllText(
        $ValidationScriptPath,
        $validationScript,
        [Text.UTF8Encoding]::new($false)
    )
    & $VenvPython $ValidationScriptPath
    Assert-LastCommandSucceeded "One or more required Python packages could not be imported."
    Remove-Item -LiteralPath $ValidationScriptPath -Force

    $modelCache = if ($env:EDGE_AI_MODEL_CACHE_DIR) {
        $env:EDGE_AI_MODEL_CACHE_DIR
    }
    else {
        Join-Path $ProjectRoot ".cache\models"
    }
    $openVinoCache = if ($env:EDGE_AI_OPENVINO_CACHE_DIR) {
        $env:EDGE_AI_OPENVINO_CACHE_DIR
    }
    else {
        Join-Path $ProjectRoot ".cache\openvino"
    }
    New-Item -ItemType Directory -Force -Path $modelCache, $openVinoCache | Out-Null

    $manifestHash = (Get-FileHash -LiteralPath $ProjectManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $markerContent = "1:$manifestHash"
    [IO.File]::WriteAllText(
        $SetupMarkerTemporary,
        $markerContent,
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $SetupMarkerTemporary -Destination $SetupMarker -Force

    Write-Host ""
    Write-Host "Setup complete."
    Write-Host "Model cache: $modelCache"
    Write-Host "OpenVINO cache: $openVinoCache"
    Write-Host "Run the application with:"
    Write-Host "  .\scripts\run.ps1"
}
catch {
    if (Test-Path -LiteralPath $SetupMarker) {
        Remove-Item -LiteralPath $SetupMarker -Force
    }
    if (Test-Path -LiteralPath $SetupMarkerTemporary) {
        Remove-Item -LiteralPath $SetupMarkerTemporary -Force
    }
    if (Test-Path -LiteralPath $ValidationScriptPath) {
        Remove-Item -LiteralPath $ValidationScriptPath -Force
    }
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
