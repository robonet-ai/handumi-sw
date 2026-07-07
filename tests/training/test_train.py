import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from handumi.training.train import _config_flags, _latest_dataset


def _make_dataset(root: Path, name: str) -> Path:
    d = root / name / "meta"
    d.mkdir(parents=True)
    (d / "info.json").write_text("{}")
    return root / name


class LatestDatasetTest(unittest.TestCase):
    def test_picks_newest_timestamp_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dataset(root, "20260101_000000")
            newest = _make_dataset(root, "20260707_120000")
            (root / "not_a_dataset").mkdir()
            self.assertEqual(_latest_dataset(root), newest)

    def test_errors_when_empty_or_missing(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                _latest_dataset(Path(tmp))
            with self.assertRaises(SystemExit):
                _latest_dataset(Path(tmp) / "does_not_exist")


class ConfigFlagsTest(unittest.TestCase):
    def test_flattens_dotted_keys_and_bools(self):
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "act.yaml"
            cfg.write_text("policy.type: act\nwandb.enable: true\nsteps: 100\n")
            self.assertEqual(
                _config_flags(cfg),
                ["--policy.type=act", "--wandb.enable=true", "--steps=100"],
            )

    def test_missing_config_errors(self):
        with self.assertRaises(SystemExit):
            _config_flags(Path("/nonexistent/act.yaml"))


if __name__ == "__main__":
    unittest.main()
