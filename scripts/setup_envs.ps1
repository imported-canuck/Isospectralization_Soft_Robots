$ErrorActionPreference = "Stop"

function New-Or-Update-Venv {
  param(
    [Parameter(Mandatory=$true)][string]$ProjectDir,
    [Parameter(Mandatory=$true)][string]$VenvName,
    [Parameter(Mandatory=$true)][string]$PythonVersion,
    [Parameter(Mandatory=$true)][string]$LockFile
  )

  Write-Host "==> $ProjectDir : $VenvName (Python $PythonVersion)"

  Push-Location $ProjectDir

  if (!(Test-Path $VenvName)) {
    Write-Host "Creating venv $VenvName ..."
    # Use the requested Python if it's on PATH (py launcher is best on Windows)
    py -$PythonVersion -m venv $VenvName
  }

  Write-Host "Installing from lockfile ..."
  & ".\$VenvName\Scripts\python.exe" -m pip install -U pip
  & ".\$VenvName\Scripts\python.exe" -m pip install -r $LockFile

  Pop-Location
}

$repo = Split-Path -Parent $PSScriptRoot
$locks = Join-Path $repo "env-locks"

New-Or-Update-Venv `
  -ProjectDir (Join-Path $repo "external_isospectralization") `
  -VenvName "tf-cpu" `
  -PythonVersion "3.6" `
  -LockFile (Join-Path $locks "requirements.tf-cpu.lock.txt")

New-Or-Update-Venv `
  -ProjectDir (Join-Path $repo "external_isospectralization_pytorch") `
  -VenvName "pt-gpu" `
  -PythonVersion "3.11" `
  -LockFile (Join-Path $locks "requirements.pt-gpu.lock.txt")

Write-Host "Done."
