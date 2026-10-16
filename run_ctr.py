"""Run independent role and company experiments with shared CLI options."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    shared_args = sys.argv[1:]
    for task in ("role", "company"):
        command = [
            sys.executable,
            str(HERE / "main_ctr.py"),
            "--prediction_task",
            task,
            *shared_args,
        ]
        print("Running", task, "model")
        subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
