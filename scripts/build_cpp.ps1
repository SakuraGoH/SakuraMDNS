# Build the native MCMC sampler DLL with MSVC + OpenMP.
#
#   pwsh scripts/build_cpp.ps1 [-Force]
#
# Output: mdns/samplers/build/mdns_samplers.dll
#
# NOTE: this script deliberately never captures an external program's stdout
# through a pipe (which some sandboxes forbid).  Discovery is done purely on
# the filesystem and `cl` writes into a log file that we read afterwards.
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "mdns\samplers\cpp\samplers.cpp"
$outDir = Join-Path $root "mdns\samplers\build"
$dll = Join-Path $outDir "mdns_samplers.dll"
$obj = Join-Path $outDir "samplers.obj"
$log = Join-Path $outDir "build.log"

if (-not (Test-Path $src)) { throw "source not found: $src" }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ((Test-Path $dll) -and (-not $Force) -and ((Get-Item $dll).LastWriteTime -ge (Get-Item $src).LastWriteTime)) {
    Write-Output "up-to-date: $dll"
    return
}

# ---- locate vcvars64.bat without spawning vswhere -------------------------
$known = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $known | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    $found = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue |
             Select-Object -First 1 -ExpandProperty FullName
    $vcvars = $found
}
if (-not $vcvars) { throw "vcvars64.bat not found; install the Visual Studio C++ build tools" }
Write-Output "using: $vcvars"

# ---- compile --------------------------------------------------------------
$bat = Join-Path $outDir "build_cmd.bat"
$lines = @(
    "@echo off",
    "call `"$vcvars`" >nul 2>&1",
    "if errorlevel 1 exit /b 2",
    "cl /nologo /O2 /openmp /std:c++17 /EHsc /MD /LD /D_CRT_SECURE_NO_WARNINGS /Fo:`"$obj`" `"$src`" /Fe:`"$dll`" > `"$log`" 2>&1",
    "exit /b %errorlevel%"
)
Set-Content -Path $bat -Encoding ASCII -Value $lines

cmd /c "`"$bat`" >nul 2>&1"
$code = $LASTEXITCODE

if (Test-Path $log) {
    $errs = Get-Content $log | Where-Object { $_ -match '\berror\b|warning C' }
    if ($errs) { $errs | Select-Object -First 15 | ForEach-Object { Write-Output $_ } }
}

if ($code -ne 0 -or -not (Test-Path $dll)) {
    if (Test-Path $log) { Get-Content $log -Tail 25 | ForEach-Object { Write-Output $_ } }
    throw "build failed (exit $code)"
}

Write-Output ("built: {0} ({1:N1} KB)" -f $dll, ((Get-Item $dll).Length / 1KB))
