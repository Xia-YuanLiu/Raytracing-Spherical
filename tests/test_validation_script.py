import json
import subprocess
import sys
from pathlib import Path


def test_validation_script_outputs_json_table(tmp_path):
    output = tmp_path / "schwarzschild_transfer.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_validation_tables.py",
            "--output",
            str(output),
            "--samples",
            "4",
        ],
        check=True,
    )

    data = json.loads(output.read_text())
    assert data["metric"] == "schwarzschild"
    assert len(data["samples"]) == 4
    assert {"b", "near_critical", "termination_reason", "intersections"} <= set(data["samples"][0])
