# PALT edge benchmark - runs three resource-budget profiles in Docker Desktop (Windows PowerShell)
# Usage (from this folder):  powershell -ExecutionPolicy Bypass -File .\run_profiles.ps1
#   quick check:             powershell -ExecutionPolicy Bypass -File .\run_profiles.ps1 -Iters 500 -Runs 2 -Cooldown 5
param(
  [int]$Iters = 10000,
  [int]$Runs = 5,
  [int]$Cooldown = 60,
  [string]$Filter = ""
)
$ErrorActionPreference = "Stop"
$Here    = $PSScriptRoot
$Models  = Join-Path $Here "models"
$Results = Join-Path $Here "results"
New-Item -ItemType Directory -Force -Path $Models, $Results | Out-Null

if (-not (Get-ChildItem $Models -Include *.onnx, *.lgb.txt, *.xgb.json -Recurse -ErrorAction SilentlyContinue)) {
  Write-Host "No models in $Models. Copy the .onnx / .lgb.txt files and bench_inputs.npz from the Kaggle results first." -ForegroundColor Yellow
  exit 1
}

# host record (goes into the paper's setup section)
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os  = Get-CimInstance Win32_OperatingSystem
$host_info = [ordered]@{
  cpu = $cpu.Name; physical_cores = $cpu.NumberOfCores; logical_processors = $cpu.NumberOfLogicalProcessors
  ram_gb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  windows = "$($os.Caption) build $($os.BuildNumber)"
  power_scheme = (powercfg /getactivescheme) -join " "
  on_ac_power = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline
  docker = (docker version --format "{{.Server.Version}}")
  started = (Get-Date -Format s)
}
$host_info | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $Results "host.json")
if ($host_info.on_ac_power -eq $false) {
  Write-Host "WARNING: running on battery. Plug in the AC adapter and set the performance power mode before the main measurement." -ForegroundColor Yellow
  if ($Iters -ge 5000) { $ans = Read-Host "Continue anyway? (y/N)"; if ($ans -ne "y") { exit 1 } }
}
Write-Host ($host_info | Out-String)

Write-Host "Building image palt-edge-bench ..." -ForegroundColor Cyan
docker build -t palt-edge-bench $Here
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

$profiles = @(
  @{ Name = "OrinNano-class"; Cpus = 6; CpuSet = "0-5"; Mem = "4g" },
  @{ Name = "RPi4-class";     Cpus = 4; CpuSet = "0-3"; Mem = "2g" },
  @{ Name = "Gateway-class";  Cpus = 2; CpuSet = "0-1"; Mem = "2g" }
)
foreach ($p in $profiles) {
  Write-Host "`n=== $($p.Name): $($p.Cpus) CPUs, $($p.Mem) RAM (no swap) ===" -ForegroundColor Cyan
  $base = @("run", "--rm", "--cpus=$($p.Cpus)", "--cpuset-cpus=$($p.CpuSet)", "--memory=$($p.Mem)", "--memory-swap=$($p.Mem)",
            "-v", "${Models}:/models:ro", "-v", "${Results}:/results",
            "palt-edge-bench", "--profile", $p.Name, "--threads", "$($p.Cpus),1", "--iters", "$Iters", "--runs", "$Runs")
  # pass 1: all models except the (very large) random forest
  $args_ = $base + @("--exclude", "RandomForest")
  if ($Filter -ne "") { $args_ += @("--filter", $Filter) }
  & docker @args_
  if ($LASTEXITCODE -ne 0) { Write-Host "profile $($p.Name) failed" -ForegroundColor Red }
  # pass 2: random forest alone, in its own container (it may exceed the memory budget; that is a result too)
  if ((Get-ChildItem $Models -Filter "*RandomForest*" -ErrorAction SilentlyContinue) -and ($Filter -eq "" -or "RandomForest" -like "*$Filter*")) {
    $args_ = $base + @("--filter", "RandomForest", "--suffix", "_rf")
    & docker @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "random forest in $($p.Name) failed (exit $LASTEXITCODE; 137 = out of memory)" -ForegroundColor Yellow }
  }
  Write-Host "cooling down $Cooldown s ..."; Start-Sleep -Seconds $Cooldown
}
Write-Host "`nDone. Share the files in $Results (edge_*.json, host.json)." -ForegroundColor Green
