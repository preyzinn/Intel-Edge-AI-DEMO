param(
    [int]$ApiPort = 8000,
    [int]$StreamlitPort = 8501
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$RequirementsFile = Join-Path $Root "requirements.txt"
$Python = $null
$LogDir = Join-Path $Root ".logs"
$script:Processes = @()
$script:Stopping = $false

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-LauncherLine {
    param([string]$Message)
    [Console]::Out.WriteLine($Message)
}

function Test-PortInUse {
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Test-HttpEndpoint {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 2
    )

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Wait-HttpEndpoint {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpEndpoint -Url $Url) {
            return $true
        }

        Start-Sleep -Seconds 1
    }

    return $false
}

function Test-CommandAvailable {
    param([string]$Command)

    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Find-SystemPython {
    $candidates = @(
        @("py", @("-3")),
        @("python", @()),
        @("python3", @())
    )

    foreach ($candidate in $candidates) {
        $command = $candidate[0]
        $prefixArgs = $candidate[1]
        if (-not (Test-CommandAvailable -Command $command)) {
            continue
        }

        $args = @()
        $args += $prefixArgs
        $args += @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)")
        try {
            & $command @args | Out-Null
            if ($LASTEXITCODE -eq 0) {
                if ($prefixArgs.Count -gt 0) {
                    return "$command $($prefixArgs -join ' ')"
                }

                return $command
            }
        }
        catch {
        }
    }

    throw "Python 3.10+ nao encontrado. Instale Python em https://www.python.org/downloads/ e marque 'Add python.exe to PATH'."
}

function Invoke-Python {
    param(
        [string]$PythonCommand,
        [string[]]$Arguments
    )

    $parts = $PythonCommand -split " "
    $command = $parts[0]
    $prefixArgs = @()
    if ($parts.Count -gt 1) {
        $prefixArgs = $parts[1..($parts.Count - 1)]
    }

    & $command @prefixArgs @Arguments
}

function Ensure-VirtualEnvironment {
    if (Test-Path $VenvPython) {
        $script:Python = $VenvPython
        return
    }

    Write-LauncherLine "Ambiente Python .venv nao encontrado. Criando..."
    $systemPython = Find-SystemPython
    Invoke-Python -PythonCommand $systemPython -Arguments @("-m", "venv", ".venv")

    if (-not (Test-Path $VenvPython)) {
        throw "Falha ao criar .venv. Verifique se o modulo venv esta disponivel no Python instalado."
    }

    $script:Python = $VenvPython
}

function Repair-StalePipMetadata {
    $sitePackages = Join-Path $Root ".venv\Lib\site-packages"
    if (-not (Test-Path $sitePackages)) {
        return
    }

    $staleItems = Get-ChildItem -LiteralPath $sitePackages -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "~ip*" }
    foreach ($item in $staleItems) {
        try {
            Write-LauncherLine "Removendo metadata temporaria antiga do pip: $($item.Name)"
            Remove-Item -LiteralPath $item.FullName -Recurse -Force -ErrorAction Stop
        }
        catch {
            Write-LauncherLine "Aviso: nao foi possivel remover $($item.Name): $($_.Exception.Message)"
        }
    }
}

function Test-PythonImports {
    param([string[]]$Modules)

    $script = @'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print(chr(44).join(missing))
    raise SystemExit(1)
'@

    $result = & $Python -c $script @Modules 2>&1
    if ($LASTEXITCODE -eq 0) {
        return @()
    }

    $missingLine = ($result | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($missingLine)) {
        return $Modules
    }

    return $missingLine.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
}

function Ensure-PythonDependencies {
    $requiredModules = @(
        "fastapi",
        "uvicorn",
        "streamlit",
        "requests",
        "pydantic"
    )

    Repair-StalePipMetadata

    Write-LauncherLine "Verificando pip..."
    & $Python -m ensurepip --upgrade | ForEach-Object { Write-LauncherLine "[setup] $_" }
    if ($LASTEXITCODE -ne 0) {
        throw "Nao foi possivel preparar pip no ambiente virtual."
    }

    Write-LauncherLine "Verificando dependencias Python..."
    $missing = Test-PythonImports -Modules $requiredModules
    if ($missing.Count -gt 0) {
        Write-LauncherLine "Dependencias ausentes: $($missing -join ', ')"
        if (-not (Test-Path $RequirementsFile)) {
            throw "Arquivo requirements.txt nao encontrado em $RequirementsFile"
        }

        Write-LauncherLine "Baixando/instalando dependencias. Isso pode demorar na primeira execucao..."
        & $Python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao atualizar pip."
        }

        & $Python -m pip install -r $RequirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao instalar dependencias de requirements.txt."
        }
    }
    else {
        Write-LauncherLine "Dependencias Python ja instaladas."
    }

    Write-LauncherLine "Validando ambiente Python..."
    & $Python -m pip check | ForEach-Object { Write-LauncherLine "[pip check] $_" }
    if ($LASTEXITCODE -ne 0) {
        throw "pip check encontrou conflitos de dependencias. Veja as mensagens acima."
    }
}

function Show-OptionalToolStatus {
    if (Test-CommandAvailable -Command "ollama") {
        Write-LauncherLine "Ollama detectado."
    }
    else {
        Write-LauncherLine "Aviso: Ollama nao encontrado. Apenas o runtime Ollama ficara indisponivel; OpenVINO/Transformers continuam funcionando."
    }
}

function Get-PortOwnerProcess {
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $connection) {
            return $null
        }

        return [System.Diagnostics.Process]::GetProcessById($connection.OwningProcess)
    }
    catch {
        return $null
    }
}

function Get-ProcessCommandLine {
    param([int]$ProcessId)

    try {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
        if ($null -eq $processInfo) {
            return ""
        }

        return [string]$processInfo.CommandLine
    }
    catch {
        return ""
    }
}

function Stop-ProcessTree {
    param([System.Diagnostics.Process]$Process)

    if ($null -eq $Process) {
        return
    }

    try {
        if (-not $Process.HasExited) {
            Write-LauncherLine "Parando processo $($Process.Id)..."
            & taskkill.exe /PID $Process.Id /T /F | ForEach-Object { Write-LauncherLine $_ }
        }
    }
    catch {
        Write-LauncherLine "Nao foi possivel parar PID $($Process.Id): $($_.Exception.Message)"
    }
}

function Stop-KnownServiceOnPort {
    param(
        [int]$Port,
        [string]$ServiceName,
        [string[]]$RequiredPatterns
    )

    $owner = Get-PortOwnerProcess -Port $Port
    if ($null -eq $owner) {
        return $false
    }

    $commandLine = Get-ProcessCommandLine -ProcessId $owner.Id
    $isKnownService = $true
    foreach ($pattern in $RequiredPatterns) {
        if ($commandLine -notlike "*$pattern*") {
            $isKnownService = $false
            break
        }
    }

    if (-not $isKnownService) {
        Write-LauncherLine "$ServiceName usa a porta $Port, mas nao parece ser deste app. PID: $($owner.Id)"
        Write-LauncherLine "Comando: $commandLine"
        return $false
    }

    Write-LauncherLine "$ServiceName antigo encontrado na porta $Port. Reiniciando sob controle deste launcher..."
    Stop-ProcessTree -Process $owner
    Start-Sleep -Seconds 2
    return -not (Test-PortInUse -Port $Port)
}

function Stop-AllServices {
    $script:Stopping = $true
    foreach ($process in $script:Processes) {
        Stop-ProcessTree -Process $process
    }
}

function ConvertTo-ArgumentString {
    param([string[]]$Arguments)

    return ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_.Replace('\', '\\').Replace('"', '\"')) + '"'
        }
        else {
            $_
        }
    }) -join " "
}

function Start-PythonService {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$OutLog,
        [string]$ErrLog
    )

    Add-Content -Path $OutLog -Value ""
    Add-Content -Path $OutLog -Value "=== Sessao iniciada em $(Get-Date -Format s) ==="
    Add-Content -Path $ErrLog -Value ""
    Add-Content -Path $ErrLog -Value "=== Sessao iniciada em $(Get-Date -Format s) ==="

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Python
    $psi.Arguments = ConvertTo-ArgumentString -Arguments $Arguments
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1"

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $process.EnableRaisingEvents = $true

    $outAction = {
        if ($EventArgs.Data) {
            $line = "[$($Event.MessageData.Name)] $($EventArgs.Data)"
            [Console]::Out.WriteLine($line)
            try {
                Add-Content -Path $Event.MessageData.Log -Value $EventArgs.Data
            }
            catch {
            }
        }
    }
    $errAction = {
        if ($EventArgs.Data) {
            $line = "[$($Event.MessageData.Name)] $($EventArgs.Data)"
            [Console]::Out.WriteLine($line)
            try {
                Add-Content -Path $Event.MessageData.Log -Value $EventArgs.Data
            }
            catch {
            }
        }
    }

    [void](Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outAction -MessageData @{ Name = $Name; Log = $OutLog })
    [void](Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errAction -MessageData @{ Name = $Name; Log = $ErrLog })

    [void]$process.Start()
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    $script:Processes += $process

    Write-LauncherLine "$Name iniciado. PID: $($process.Id)"
    return $process
}

$cancelHandler = [ConsoleCancelEventHandler]{
    param($sender, $eventArgs)
    $eventArgs.Cancel = $true
    Write-LauncherLine ""
    Write-LauncherLine "Ctrl+C recebido. Encerrando FastAPI e Streamlit..."
    Stop-AllServices
}
[Console]::add_CancelKeyPress($cancelHandler)

try {
    Write-LauncherLine "Intel Edge AI Demo"
    Write-LauncherLine "Pasta: $Root"
    Write-LauncherLine ""

    Ensure-VirtualEnvironment
    Write-LauncherLine "Python: $Python"
    Ensure-PythonDependencies
    Show-OptionalToolStatus
    Write-LauncherLine ""

    $ApiManaged = $false
    $StreamlitManaged = $false

    if ((Test-PortInUse -Port $ApiPort) -and -not (Stop-KnownServiceOnPort -Port $ApiPort -ServiceName "FastAPI" -RequiredPatterns @("uvicorn", "main:app"))) {
        Write-LauncherLine "FastAPI ja esta rodando em http://127.0.0.1:$ApiPort"
        Write-LauncherLine "Aviso: processo existente nessa porta nao sera parado por este launcher."
    }
    else {
        $ApiProcess = Start-PythonService `
            -Name "uvicorn" `
            -Arguments @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
            -OutLog (Join-Path $LogDir "uvicorn-$ApiPort.out.log") `
            -ErrLog (Join-Path $LogDir "uvicorn-$ApiPort.err.log")
        $ApiManaged = $true
    }

    if (Test-PortInUse -Port $StreamlitPort) {
        if (Stop-KnownServiceOnPort -Port $StreamlitPort -ServiceName "Streamlit" -RequiredPatterns @("streamlit", "View/app.py")) {
            $StreamlitProcess = Start-PythonService `
                -Name "streamlit" `
                -Arguments @("-m", "streamlit", "run", "View/app.py", "--server.address", "127.0.0.1", "--server.port", "$StreamlitPort") `
                -OutLog (Join-Path $LogDir "streamlit-$StreamlitPort.out.log") `
                -ErrLog (Join-Path $LogDir "streamlit-$StreamlitPort.err.log")
            $StreamlitManaged = $true
        }
        else {
            throw "A porta $StreamlitPort esta ocupada por outro processo. Feche esse processo ou rode: .\run_app.ps1 -StreamlitPort 8502"
        }
    }
    else {
        $StreamlitProcess = Start-PythonService `
            -Name "streamlit" `
            -Arguments @("-m", "streamlit", "run", "View/app.py", "--server.address", "127.0.0.1", "--server.port", "$StreamlitPort") `
            -OutLog (Join-Path $LogDir "streamlit-$StreamlitPort.out.log") `
            -ErrLog (Join-Path $LogDir "streamlit-$StreamlitPort.err.log")
        $StreamlitManaged = $true
    }

    $StreamlitUrl = "http://127.0.0.1:$StreamlitPort"
    if (-not (Wait-HttpEndpoint -Url $StreamlitUrl -TimeoutSeconds 30)) {
        Write-LauncherLine ""
        Write-LauncherLine "Streamlit nao respondeu em $StreamlitUrl."
        Write-LauncherLine "Veja os ultimos logs:"
        $streamlitErrLog = Join-Path $LogDir "streamlit-$StreamlitPort.err.log"
        $streamlitOutLog = Join-Path $LogDir "streamlit-$StreamlitPort.out.log"
        if (Test-Path $streamlitErrLog) {
            Get-Content $streamlitErrLog -Tail 30 | ForEach-Object { Write-LauncherLine "[streamlit err] $_" }
        }
        if (Test-Path $streamlitOutLog) {
            Get-Content $streamlitOutLog -Tail 30 | ForEach-Object { Write-LauncherLine "[streamlit out] $_" }
        }
        throw "Streamlit nao iniciou corretamente na porta $StreamlitPort."
    }

    Write-LauncherLine ""
    Write-LauncherLine "Link do Streamlit:"
    Write-LauncherLine $StreamlitUrl
    Write-LauncherLine ""
    Write-LauncherLine "Logs tambem salvos em:"
    Write-LauncherLine "- $LogDir\uvicorn-$ApiPort.out.log"
    Write-LauncherLine "- $LogDir\uvicorn-$ApiPort.err.log"
    Write-LauncherLine "- $LogDir\streamlit-$StreamlitPort.out.log"
    Write-LauncherLine "- $LogDir\streamlit-$StreamlitPort.err.log"
    Write-LauncherLine ""
    Write-LauncherLine "Deixe esta janela aberta. Pressione Ctrl+C para parar tudo que foi iniciado por este launcher."
    Write-LauncherLine ""

    while (-not $script:Stopping) {
        foreach ($process in $script:Processes) {
            if ($process.HasExited) {
                Write-LauncherLine "Processo PID $($process.Id) terminou com codigo $($process.ExitCode). Encerrando os demais."
                Stop-AllServices
                exit $process.ExitCode
            }
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    [Console]::remove_CancelKeyPress($cancelHandler)
    Stop-AllServices
    Write-LauncherLine "Aplicacao encerrada."
}
