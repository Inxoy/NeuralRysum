# Export the newest checkpoint of a run to a .nrys file for the plugin.
#     .\scripts\export_latest.ps1                # once
#     .\scripts\export_latest.ps1 -EveryMinutes 30 -WhilePid 1234   # repeat while that process lives
# Works on a copy of last.pt, runs on the CPU, and replaces the .nrys only when
# the export succeeded, so it is safe to run while training.
param(
    [string]$Run = "runs\rysum",
    [string]$Data = "data\mariiaiii_48k_mono",
    [string]$Out = "models\rysum_latest.nrys",
    [int]$EveryMinutes = 0,
    [int]$WhilePid = 0
)
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Split-Path $Out -Parent) | Out-Null
$env:CUDA_VISIBLE_DEVICES = ""
$last = ""
while ($true) {
    $ck = Get-Item (Join-Path $Run "last.pt") -ErrorAction SilentlyContinue
    if ($ck -and $ck.LastWriteTime.Ticks -ne $last) {
        $tmpCk = Join-Path $Run "export_copy.pt"
        $tmpOut = "$Out.tmp"
        try {
            Copy-Item $ck.FullName $tmpCk -Force -ErrorAction Stop
            & $py -m neuralrysum.export --ckpt $tmpCk --out $tmpOut --data $Data *> (Join-Path $Run "export.log")
            if ($LASTEXITCODE -eq 0 -and (Test-Path $tmpOut)) {
                Move-Item $tmpOut $Out -Force
                $step = & $py -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu')['step'])" $tmpCk
                "$(Get-Date -Format 'yyyy-MM-dd HH:mm') step $step" | Out-File "$Out.txt" -Encoding ascii
                $last = $ck.LastWriteTime.Ticks
            }
        } catch {}
    }
    if ($EveryMinutes -le 0) { break }
    if ($WhilePid -gt 0 -and -not (Get-Process -Id $WhilePid -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Seconds ($EveryMinutes * 60)
}
