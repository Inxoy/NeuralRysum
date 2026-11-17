# Dot-source this to put the local MSVC toolchain (.toolchain, see fetch_msvc.py),
# CMake and Ninja on PATH for the current PowerShell session:
#     . .\scripts\msvc_env.ps1
$repo = Split-Path $PSScriptRoot -Parent
$e = Get-Content (Join-Path $repo ".toolchain\env.json") -Raw | ConvertFrom-Json
$env:INCLUDE = $e.INCLUDE -join ";"
$env:LIB = $e.LIB -join ";"
$env:PATH = (($e.PATH + (Join-Path $repo ".venv\Scripts")) -join ";") + ";" + $env:PATH
