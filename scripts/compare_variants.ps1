# Short spectral-only trainings of several recipes, one after another, then the leak test.
#     .\scripts\compare_variants.ps1 [-Steps 50000] [-Data data\mariiaiii_48k_mono] [-Root runs\variants]
# A variant whose last.pt already reached -Steps is skipped, so the script can be rerun after a stop.
param(
    [int]$Steps = 50000,
    [string]$Data = "data\mariiaiii_48k_mono",
    [string]$Root = "runs\variants",
    [string[]]$Baseline = @("runs\rysum\step_50000.pt")
)
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$variants = [ordered]@{
    "dim4"               = @("--fast_dim", "4")
    "dim4_noise"         = @("--fast_dim", "4", "--code_noise", "0.2")
    "dim12_noise"        = @("--code_noise", "0.3")
    "dim12_strong"       = @("--perturb", "strong")
    "dim4_noise_strong"  = @("--fast_dim", "4", "--code_noise", "0.2", "--perturb", "strong")
}
New-Item -ItemType Directory -Force $Root | Out-Null
$ckpts = @($Baseline)
foreach ($name in $variants.Keys) {
    $out = Join-Path $Root $name
    New-Item -ItemType Directory -Force $out | Out-Null
    "$(Get-Date -Format 'HH:mm:ss') start $name" | Out-File (Join-Path $Root "progress.txt") -Append -Encoding ascii
    & $py scripts\train_keepawake.py --data $Data --out $out --steps $Steps --save_every 10000 @($variants[$name]) `
        >> (Join-Path $out "train.log") 2>> (Join-Path $out "train.err.log")
    "$(Get-Date -Format 'HH:mm:ss') done  $name (exit $LASTEXITCODE)" | Out-File (Join-Path $Root "progress.txt") -Append -Encoding ascii
    $ckpts += (Join-Path $out "last.pt")
}
$env:CUDA_VISIBLE_DEVICES = ""
& $py scripts\leak_test.py --data $Data --ckpt $ckpts --detail --wav-dir (Join-Path $Root "leak_wav") `
    --json (Join-Path $Root "leak.json") > (Join-Path $Root "leak.txt")
"$(Get-Date -Format 'HH:mm:ss') leak test done" | Out-File (Join-Path $Root "progress.txt") -Append -Encoding ascii
