using System;
using System.Diagnostics;
using System.IO;

class Program
{
    static Process powershellProcess;
    static bool stopping = false;

    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(root, "run_app.ps1");

        Console.Title = "Intel Edge AI Demo";
        Console.WriteLine("Iniciando Intel Edge AI Demo...");
        Console.WriteLine("Launcher: " + System.Reflection.Assembly.GetExecutingAssembly().Location);
        Console.WriteLine("Script: " + script);
        Console.WriteLine("Deixe esta janela aberta para ver logs, downloads e erros.");
        Console.WriteLine("Quando iniciar corretamente, abra: http://127.0.0.1:8501");
        Console.WriteLine("Pressione Ctrl+C para parar FastAPI, Streamlit e processos filhos.");
        Console.WriteLine();

        if (!File.Exists(script))
        {
            Console.Error.WriteLine("Nao encontrei o arquivo run_app.ps1 em:");
            Console.Error.WriteLine(script);
            WaitBeforeClose();
            return 1;
        }

        Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            if (stopping) return;

            stopping = true;
            Console.WriteLine();
            Console.WriteLine("Ctrl+C recebido. Parando a aplicacao inteira...");
            KillProcessTree(powershellProcess);
        };

        var psi = new ProcessStartInfo();
        psi.FileName = "powershell.exe";
        psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\"";
        psi.WorkingDirectory = root;
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        powershellProcess = new Process();
        powershellProcess.StartInfo = psi;
        powershellProcess.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null) Console.WriteLine(e.Data);
        };
        powershellProcess.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null) Console.Error.WriteLine(e.Data);
        };

        try
        {
            powershellProcess.Start();
            powershellProcess.BeginOutputReadLine();
            powershellProcess.BeginErrorReadLine();
            powershellProcess.WaitForExit();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Falha ao iniciar o PowerShell: " + ex.Message);
            WaitBeforeClose();
            return 1;
        }

        if (stopping)
        {
            Console.WriteLine("Aplicacao encerrada por Ctrl+C.");
            return 130;
        }

        Console.WriteLine();
        if (powershellProcess.ExitCode != 0)
        {
            Console.WriteLine("Ocorreu um erro ao iniciar ou manter a aplicacao. Codigo: " + powershellProcess.ExitCode);
            Console.WriteLine("Verifique estes logs:");
            Console.WriteLine(Path.Combine(root, ".logs", "streamlit-8501.err.log"));
            Console.WriteLine(Path.Combine(root, ".logs", "streamlit-8501.out.log"));
        }
        else
        {
            Console.WriteLine("Aplicacao encerrada.");
        }

        WaitBeforeClose();
        return powershellProcess.ExitCode;
    }

    static void KillProcessTree(Process process)
    {
        if (process == null) return;

        try
        {
            if (process.HasExited) return;

            var taskkillInfo = new ProcessStartInfo();
            taskkillInfo.FileName = "taskkill.exe";
            taskkillInfo.Arguments = "/PID " + process.Id + " /T /F";
            taskkillInfo.UseShellExecute = false;
            taskkillInfo.RedirectStandardOutput = true;
            taskkillInfo.RedirectStandardError = true;
            taskkillInfo.CreateNoWindow = true;

            using (var taskkill = Process.Start(taskkillInfo))
            {
                string output = taskkill.StandardOutput.ReadToEnd();
                string error = taskkill.StandardError.ReadToEnd();
                taskkill.WaitForExit();

                if (!String.IsNullOrWhiteSpace(output)) Console.WriteLine(output.Trim());
                if (!String.IsNullOrWhiteSpace(error)) Console.Error.WriteLine(error.Trim());
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Nao foi possivel parar a arvore de processos: " + ex.Message);
        }
    }

    static void WaitBeforeClose()
    {
        Console.WriteLine();
        Console.WriteLine("Pressione qualquer tecla para fechar esta janela.");
        Console.ReadKey(true);
    }
}
