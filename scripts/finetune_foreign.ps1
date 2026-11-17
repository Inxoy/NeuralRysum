# Fine-tune a finished run with the foreign-input transfer objective, then score the snapshots.
#     .\scripts\finetune_foreign.ps1 [-From runs\rysum\last.pt] [-Out runs\foreign] [-Steps 40000]
# The source run is only read. Rerunning resumes from <Out>\last.pt.
param(
    [string]$From = "runs\rysum\last.pt",
    [string]$Out = "runs\foreign",
    [int]$Steps = 40000,
    [string]$Data = "data\mariiaiii_48k_mono",
    [string]$ForeignWeight = "1.0"
)
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Out | Out-Null
$last = Join-Path $Out "last.pt"
if (-not (Test-Path $last)) { Copy-Item $From $last }
$start = [int](& $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu')['step'])" $From)
& $py scripts\train_keepawake.py --data $Data --out $Out --steps ($start + $Steps) --save_every 1000 `
    --foreign_weight $ForeignWeight >> (Join-Path $Out "train.log") 2>> (Join-Path $Out "train.err.log")
$env:CUDA_VISIBLE_DEVICES = ""
$ckpts = @($From) + @(Get-ChildItem $Out -Filter "step_*.pt" | Sort-Object { [int]($_.BaseName -replace 'step_', '') } | ForEach-Object { $_.FullName })
& $py scripts\leak_test.py --data $Data --ckpt $ckpts --detail --wav-dir (Join-Path $Out "leak_wav") `
    --json (Join-Path $Out "leak.json") > (Join-Path $Out "leak.txt")
