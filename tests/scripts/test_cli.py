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
    assert "completion" in output


def test_dispatch_forwards_remaining_arguments():
    target_main = mock.Mock()
    module = SimpleNamespace(main=target_main)
    with mock.patch.object(cli.importlib, "import_module", return_value=module):
        cli.main(["doctor", "--strict"])

    target_main.assert_called_once_with()


def test_nested_command_routes_to_specialized_module():
    target_main = mock.Mock()
    module = SimpleNamespace(main=target_main)
    with mock.patch.object(cli.importlib, "import_module", return_value=module) as load:
        cli.main(["calibrate", "spatial", "verify"])

    load.assert_called_once_with("handumi.scripts.setup.calibrate_spatial")
    target_main.assert_called_once_with()


def test_command_group_help_lists_only_its_subcommands(capsys):
    cli.main(["teleop", "--help"])

    output = capsys.readouterr().out
    assert "handumi teleop sim" in output
    assert "handumi teleop real" in output
    assert "handumi calibrate spatial" not in output


def test_short_program_name_is_preserved_in_help(capsys):
    with mock.patch.object(cli.sys, "argv", ["hu"]):
        cli.main(["teleop", "--help"])

    output = capsys.readouterr().out
    assert "hu teleop sim" in output
    assert "handumi teleop sim" not in output


def test_unknown_command_fails_cleanly():
    with pytest.raises(SystemExit, match="Unknown HandUMI command"):
        cli.main(["unknown"])
