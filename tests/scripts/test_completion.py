from unittest import mock

from handumi.scripts import completion


def test_completes_root_commands():
    assert completion.completion_candidates(["re"]) == ["record", "replay"]


def test_completes_nested_commands():
    assert completion.completion_candidates(["teleop", ""]) == ["real", "sim"]
    assert completion.completion_candidates(["calibrate", "sp"]) == ["spatial"]


def test_completes_public_shell_names():
    assert completion.completion_candidates(["completion", "z"]) == ["zsh"]


def test_completes_command_options():
    with mock.patch.object(
        completion,
        "_command_help",
        return_value="options:\n  --device {pico,meta}\n  --dry-run\n",
    ):
        assert completion.completion_candidates(["record", "--d"]) == [
            "--device",
            "--dry-run",
        ]
        assert completion.completion_candidates(
            ["record", "--device", "m"]
        ) == ["meta"]


def test_generated_scripts_call_dynamic_candidate_protocol():
    for script in completion.SHELL_COMPLETIONS.values():
        assert "completion __complete" in script
        assert "handumi" in script
        assert "hu" in script
