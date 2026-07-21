from types import SimpleNamespace
from unittest import mock

import pytest

from handumi.scripts import cli


def test_help_lists_common_workflow(capsys):
    cli.main([])

    output = capsys.readouterr().out
    assert "handumi doctor" in output
    assert "record" in output
    assert "convert" in output


def test_dispatch_forwards_remaining_arguments():
    target_main = mock.Mock()
    module = SimpleNamespace(main=target_main)
    with mock.patch.object(cli.importlib, "import_module", return_value=module):
        cli.main(["doctor", "--strict"])

    target_main.assert_called_once_with()


def test_unknown_command_fails_cleanly():
    with pytest.raises(SystemExit, match="Unknown HandUMI command"):
        cli.main(["unknown"])
