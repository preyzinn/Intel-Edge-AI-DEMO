$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $ProjectRoot "launcher\StartEdgeAIDemo.cs"
$Output = Join-Path $ProjectRoot "Start_Edge_AI.exe"
$TemporaryOutput = Join-Path $ProjectRoot "Start_Edge_AI.build.exe"
$CompilerCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)

try {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Launcher source not found: $Source"
    }

    $Compiler = $CompilerCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $Compiler) {
        throw "The built-in .NET Framework C# compiler was not found. Enable .NET Framework 4.x and try again."
    }

    if (Test-Path -LiteralPath $TemporaryOutput) {
        Remove-Item -LiteralPath $TemporaryOutput -Force
    }

    Write-Host "Building Windows launcher..."
    $CompilerArguments = @(
        "/nologo",
        "/target:exe",
        "/platform:anycpu",
        "/optimize+",
        "/debug-",
        "/out:$TemporaryOutput",
        $Source
    )
    & $Compiler @CompilerArguments
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $TemporaryOutput)) {
        throw "The launcher compiler failed with exit code $LASTEXITCODE."
    }

    Move-Item -LiteralPath $TemporaryOutput -Destination $Output -Force
    Write-Host "Launcher created: $Output"
    Write-Host "Double-click Start_Edge_AI.exe to set up and run the project."
}
catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
finally {
    if (Test-Path -LiteralPath $TemporaryOutput) {
        Remove-Item -LiteralPath $TemporaryOutput -Force
    }
}
