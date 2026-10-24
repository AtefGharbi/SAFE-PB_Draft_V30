$ErrorActionPreference = "Stop"
Write-Host "SAFE-PB convergence study: 3 seeds x budgets 300,500,750,1000"
python scripts/run_convergence_study.py --budgets 300,500,750,1000 --seeds 0,1,2 --eval-days 10 --replay-capacity 5000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Finished. Open results/convergence_study/convergence_summary.csv"
