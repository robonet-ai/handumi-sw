import json
import subprocess
from pathlib import Path


def test_release_inventory_maps_every_robot_asset_and_source_dependency():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        ["python", "bin/verify_release_inventory.py", "--root", str(root)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["asset_sets"] == 6
    assert result["mapped_files"] > 100
    assert result["dependencies"] >= 4
