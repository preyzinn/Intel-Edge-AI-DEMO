using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;

[assembly: AssemblyTitle("Intel Edge AI Demo Launcher")]
[assembly: AssemblyDescription("Sets up and starts the local PyTorch versus OpenVINO benchmark")]
[assembly: AssemblyCompany("Intel Edge AI Demo")]
[assembly: AssemblyProduct("Intel Edge AI Demo")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class Program
{
    private const int DefaultPort = 8501;
    private const int ReadinessTimeoutMilliseconds = 120000;

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            Console.Title = "Intel Edge AI Demo";

            int port;
            if (!TryReadPort(args, out port))
            {
                ShowError("Usage: Start_Edge_AI.exe [port] or Start_Edge_AI.exe -Port <port>");
                return 2;
            }

            string projectRoot = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
            string setupScript = Path.Combine(projectRoot, "scripts", "setup.ps1");
            string runScript = Path.Combine(projectRoot, "scripts", "run.ps1");
            string entryPoint = Path.Combine(projectRoot, "src", "edge_ai_demo", "app.py");
            string venvPython = Path.Combine(projectRoot, ".venv", "Scripts", "python.exe");
            string setupMarker = Path.Combine(projectRoot, ".venv", ".edge-ai-setup");
            string projectManifest = Path.Combine(projectRoot, "pyproject.toml");

            if (
                !File.Exists(setupScript)
                || !File.Exists(runScript)
                || !File.Exists(entryPoint)
                || !File.Exists(projectManifest)
            )
            {
                ShowError(
                    "The launcher must remain in the project root beside the scripts and src folders."
                );
                return 3;
            }

            PrintBanner(projectRoot);

            if (!IsEnvironmentReady(venvPython, setupMarker, projectManifest, projectRoot))
            {
                Console.WriteLine("Setup is missing or incomplete. Running setup now...");
                Console.WriteLine();

                int setupExitCode = RunPowerShell(setupScript, projectRoot, null);
                if (setupExitCode != 0)
                {
                    ShowError("Setup failed. Review the messages above, then run the launcher again.");
                    return setupExitCode;
                }

                if (
                    !IsEnvironmentReady(
                        venvPython,
                        setupMarker,
                        projectManifest,
                        projectRoot
                    )
                )
                {
                    ShowError("Setup finished, but the required Python environment is not ready.");
                    return 4;
                }
            }
            else
            {
                Console.WriteLine("Setup is ready; skipping installation.");
            }

            Console.WriteLine("Starting the benchmark at http://127.0.0.1:" + port);
            Console.WriteLine("Keep this window open while the application is running.");
            Console.WriteLine();

            int runExitCode = RunPowerShell(runScript, projectRoot, port);
            if (runExitCode != 0)
            {
                ShowError("The application stopped with exit code " + runExitCode + ".");
            }

            return runExitCode;
        }
        catch (Exception exception)
        {
            ShowError("Could not start the application: " + exception.Message);
            return 1;
        }
    }

    private static bool TryReadPort(string[] args, out int port)
    {
        port = DefaultPort;

        string value;
        if (args.Length == 0)
        {
            return true;
        }
        if (args.Length == 1)
        {
            value = args[0];
        }
        else if (
            args.Length == 2
            && args[0].Equals("-Port", StringComparison.OrdinalIgnoreCase)
        )
        {
            value = args[1];
        }
        else
        {
            return false;
        }

        int parsedPort;
        if (
            !int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out parsedPort)
            || parsedPort < 1
            || parsedPort > 65535
        )
        {
            return false;
        }

        port = parsedPort;
        return true;
    }

    private static bool IsEnvironmentReady(
        string pythonExecutable,
        string setupMarker,
        string projectManifest,
        string projectRoot
    )
    {
        if (
            !File.Exists(pythonExecutable)
            || !SetupMarkerMatches(setupMarker, projectManifest)
        )
        {
            return false;
        }

        const string probe =
            "import edge_ai_demo, openvino, psutil, streamlit, torch, transformers; "
            + "import optimum.intel";

        return RunQuietly(
                pythonExecutable,
                "-m pip check",
                projectRoot,
                ReadinessTimeoutMilliseconds
            )
            && RunQuietly(
                pythonExecutable,
                "-B -c \"" + probe + "\"",
                projectRoot,
                ReadinessTimeoutMilliseconds
            );
    }

    private static bool SetupMarkerMatches(string setupMarker, string projectManifest)
    {
        if (!File.Exists(setupMarker) || !File.Exists(projectManifest))
        {
            return false;
        }

        string expectedMarker;
        using (SHA256 sha256 = SHA256.Create())
        using (FileStream manifestStream = File.OpenRead(projectManifest))
        {
            string hash = BitConverter.ToString(sha256.ComputeHash(manifestStream))
                .Replace("-", string.Empty)
                .ToLowerInvariant();
            expectedMarker = "1:" + hash;
        }

        string actualMarker = File.ReadAllText(setupMarker).Trim();
        return actualMarker.Equals(expectedMarker, StringComparison.OrdinalIgnoreCase);
    }

    private static bool RunQuietly(
        string executable,
        string arguments,
        string workingDirectory,
        int timeoutMilliseconds
    )
    {

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = executable,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;
            process.OutputDataReceived += delegate { };
            process.ErrorDataReceived += delegate { };

            if (!process.Start())
            {
                return false;
            }

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            if (!process.WaitForExit(timeoutMilliseconds))
            {
                process.Kill();
                process.WaitForExit();
                return false;
            }

            process.WaitForExit();
            return process.ExitCode == 0;
        }
    }

    private static int RunPowerShell(string script, string projectRoot, int? port)
    {
        string systemDirectory = Environment.GetFolderPath(Environment.SpecialFolder.System);
        string powershell = Path.Combine(
            systemDirectory,
            "WindowsPowerShell",
            "v1.0",
            "powershell.exe"
        );
        if (!File.Exists(powershell))
        {
            powershell = "powershell.exe";
        }

        string arguments =
            "-NoLogo -NoProfile -ExecutionPolicy Bypass -File " + QuoteArgument(script);
        if (port.HasValue)
        {
            arguments += " -Port " + port.Value.ToString(CultureInfo.InvariantCulture);
        }

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = powershell,
            Arguments = arguments,
            WorkingDirectory = projectRoot,
            UseShellExecute = false
        };

        using (Process process = Process.Start(startInfo))
        {
            if (process == null)
            {
                throw new InvalidOperationException("Windows PowerShell could not be started.");
            }

            process.WaitForExit();
            return process.ExitCode;
        }
    }

    private static string QuoteArgument(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static void PrintBanner(string projectRoot)
    {
        Console.WriteLine("Intel Edge AI Demo");
        Console.WriteLine("PyTorch versus OpenVINO benchmark");
        Console.WriteLine("Project: " + projectRoot.TrimEnd(Path.DirectorySeparatorChar));
        Console.WriteLine();
    }

    private static void ShowError(string message)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("ERROR: " + message);
        Console.Error.WriteLine("Press Enter to close this window.");
        try
        {
            Console.ReadLine();
        }
        catch (IOException)
        {
            // No interactive console is attached.
        }
    }
}
