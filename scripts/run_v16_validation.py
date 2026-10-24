"""Run the ordered V16 validation and preserve a machine-readable status log."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
STATUS = os.path.join(RESULTS, "v16_validation_status.json")

COMMANDS = [
    [sys.executable, "scripts/run_network_calibration.py"],
    [sys.executable, "scripts/run_baseline_validation.py"],
    [sys.executable, "scripts/run_safety_layer_demo.py"],
    [sys.executable, "scripts/run_safety_effectiveness_demo.py"],
    [sys.executable, "scripts/run_sac_smoke_demo.py"],
    [sys.executable, "-m", "pytest", "tests", "-q"],
]


def write_status(rows, overall):
    os.makedirs(RESULTS, exist_ok=True)
    with open(STATUS, "w") as f:
        json.dump({
            "generated_by": "scripts/run_v16_validation.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall,
            "commands": rows,
        }, f, indent=2)


def main():
    rows = []
    for command in COMMANDS:
        proc = subprocess.run(command, cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        portable_command = ["python3" if x == sys.executable else x for x in command]
        portable_output = proc.stdout.replace(ROOT + os.sep, "").replace(sys.executable, "python3")
        row = {"command": portable_command, "returncode": proc.returncode,
               "output": portable_output}
        rows.append(row)
        write_status(rows, "running" if proc.returncode == 0 else "failed")
        print(proc.stdout)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    write_status(rows, "passed")


if __name__ == "__main__":
    main()
