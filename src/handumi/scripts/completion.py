"""Generate shell completion for the unified HandUMI command."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import re
import sys

from handumi.scripts.cli import COMMANDS


BASH_COMPLETION = r'''_handumi_complete() {
  local -a request
  local candidate
  local executable="${COMP_WORDS[0]}"
  request=("${COMP_WORDS[@]:1:COMP_CWORD}")
  COMPREPLY=()
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && COMPREPLY+=("$candidate")
  done < <(command "$executable" completion __complete -- "${request[@]}")
  if ((${#COMPREPLY[@]} == 0)); then
    compopt -o default 2>/dev/null || true
  fi
}
complete -o bashdefault -o default -F _handumi_complete handumi hu
'''


ZSH_COMPLETION = r'''autoload -Uz compinit
if ! (( $+functions[compdef] )); then
  compinit
fi

_handumi_complete() {
  local -a request candidates
  local executable="${words[1]}"
  request=("${words[2,-1]}")
  candidates=("${(@f)$(command "$executable" completion __complete -- "${request[@]}")}")
  if (( ${#candidates[@]} )); then
    compadd -- "${candidates[@]}"
  else
    _files
  fi
}
compdef _handumi_complete handumi hu
'''


FISH_COMPLETION = r'''function __handumi_complete
    set -l tokens (commandline -opc)
    set -l executable $tokens[1]
    set -e tokens[1]
    command $executable completion __complete -- $tokens (commandline -ct)
end
complete -c handumi -f -a '(__handumi_complete)'
complete -c hu -f -a '(__handumi_complete)'
'''


SHELL_COMPLETIONS = {
    "bash": BASH_COMPLETION,
    "zsh": ZSH_COMPLETION,
    "fish": FISH_COMPLETION,
}


def _direct_children(prefix: tuple[str, ...]) -> set[str]:
    return {
        path[len(prefix)]
        for path in COMMANDS
        if path[: len(prefix)] == prefix and len(path) > len(prefix)
    }


def _matched_command(words: list[str]) -> tuple[str, ...] | None:
    return next(
        (
            path
            for path in sorted(COMMANDS, key=len, reverse=True)
            if tuple(words[: len(path)]) == path
        ),
        None,
    )


def _invoke_help(path: tuple[str, ...], flag: str) -> str:
    command = COMMANDS[path]
    module = importlib.import_module(command.module)
    output = io.StringIO()
    previous_argv = sys.argv
    try:
        sys.argv = [f"handumi {' '.join(path)}", flag]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                module.main()
            except SystemExit:
                pass
    finally:
        sys.argv = previous_argv
    return output.getvalue()


def _command_help(path: tuple[str, ...]) -> str:
    normal_help = _invoke_help(path, "--help")
    if "--help-advanced" not in normal_help:
        return normal_help
    return _invoke_help(path, "--help-advanced")


def _option_candidates(path: tuple[str, ...], prefix: str) -> list[str]:
    help_text = _command_help(path)
    options = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", help_text))
    return sorted(option for option in options if option.startswith(prefix))


def _choice_candidates(
    path: tuple[str, ...], option: str, prefix: str
) -> list[str]:
    help_text = _command_help(path)
    match = re.search(rf"{re.escape(option)}(?:[ =])\{{([^}}]+)\}}", help_text)
    if match is None:
        return []
    return sorted(
        choice
        for choice in match.group(1).split(",")
        if choice.startswith(prefix)
    )


def completion_candidates(words: list[str]) -> list[str]:
    """Return newline-safe candidates for the current shell token."""
    current = words[-1] if words else ""
    committed = words[:-1]

    command_prefix: list[str] = []
    for token in committed:
        if token.startswith("-"):
            break
        candidate = (*command_prefix, token)
        if not any(path[: len(candidate)] == candidate for path in COMMANDS):
            break
        command_prefix.append(token)

    if len(command_prefix) == len(committed) and not current.startswith("-"):
        children = _direct_children(tuple(command_prefix))
        matches = sorted(child for child in children if child.startswith(current))
        if matches:
            return matches

    path = _matched_command(committed)
    if path is None:
        return []

    if path == ("completion",) and len(committed) == 1:
        return sorted(
            shell for shell in SHELL_COMPLETIONS if shell.startswith(current)
        )
    if current.startswith("--"):
        return _option_candidates(path, current)
    if committed and committed[-1].startswith("--"):
        return _choice_candidates(path, committed[-1], current)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print shell completion code for HandUMI."
    )
    parser.add_argument("shell", choices=tuple(SHELL_COMPLETIONS))
    return parser


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "__complete":
        words = values[1:]
        if words[:1] == ["--"]:
            words = words[1:]
        print("\n".join(completion_candidates(words)))
        return
    args = build_parser().parse_args(values)
    print(SHELL_COMPLETIONS[args.shell], end="")


if __name__ == "__main__":
    main()
