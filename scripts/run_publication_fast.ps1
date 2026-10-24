param(
    [int]$Workers = 4,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

& $Python scripts/run_v17_experiments.py --workers $Workers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
