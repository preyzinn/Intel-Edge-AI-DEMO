param(
    [int]$ApiPort = 8000,
    [int]$StreamlitPort = 8501
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
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

function Get-PortOwnerProcess {
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
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
    Write-LauncherLine "Python: $Python"
    Write-LauncherLine "Pasta: $Root"
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

    if ((Test-PortInUse -Port $StreamlitPort) -and -not (Stop-KnownServiceOnPort -Port $StreamlitPort -ServiceName "Streamlit" -RequiredPatterns @("streamlit", "View/app.py"))) {
        Write-LauncherLine "Streamlit ja esta rodando em http://localhost:$StreamlitPort"
        Write-LauncherLine "Aviso: processo existente nessa porta nao sera parado por este launcher."
    }
    else {
        $StreamlitProcess = Start-PythonService `
            -Name "streamlit" `
            -Arguments @("-m", "streamlit", "run", "View/app.py", "--server.address", "localhost", "--server.port", "$StreamlitPort") `
            -OutLog (Join-Path $LogDir "streamlit-$StreamlitPort.out.log") `
            -ErrLog (Join-Path $LogDir "streamlit-$StreamlitPort.err.log")
        $StreamlitManaged = $true
    }

    Start-Sleep -Seconds 3

    Write-LauncherLine ""
    Write-LauncherLine "Link do Streamlit:"
    Write-LauncherLine "http://localhost:$StreamlitPort"
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
