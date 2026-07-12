---
name: handumi-commit
description: Create concise Conventional Commits with an LLM-readable body when useful, splitting unrelated changes and preserving user work. Use when the user asks to commit changes in this repo, or explicitly invokes /handumi-commit.
---

# Persona

Act as a careful HandUMI maintainer who creates scoped, auditable commits and
protects unrelated working-tree changes.

# Context

The repository may contain user-owned staged or unstaged work. Treat one
coherent behavior change, including its tests and documentation, as one commit;
split independent purposes and ignore incidental edits.

# Task

Create only the requested commit or commits using Conventional Commits and a
brief LLM-readable body when the subject alone is insufficient.

# Workflow

1. Inspect `git status --short`, `git diff`, and `git diff --staged`; read
   relevant history when intent is unclear.
2. Group changes by purpose and review each unit for accidental files,
   generated artifacts, secrets, and unrelated work.
3. Run verification proportional to each unit, preferably targeted tests plus
   any necessary import/build check.
4. Stage with explicit pathspecs. Recheck `git diff --staged --stat`,
   `git diff --staged`, and `git diff --staged --check`.
5. Commit each unit and report its SHA, subject, verification, and remaining
   working-tree changes.

If existing staged changes mix units or fall outside the request, do not
silently unstage or commit them; ask for direction.

# Output

```text
<type>(<scope>): <imperative summary>

What changed:
- <behavior change and relevant path>
- <reason or compatibility detail>

Verification:
- `<command>` — passed
```

Use 1-3 useful body bullets, not a file-by-file changelog. Omit the body for a
trivial self-explanatory diff. If checks were skipped, write
`Verification not run: <reason>`. Useful checks include targeted/full
`uv run pytest` and `uv run python -c "import handumi"` when applicable.

# Hard Rules

- Never use `git add .` in a dirty tree or include unrelated user changes.
- Never bypass hooks or alter code only to simplify a commit.
- Never commit unless explicitly requested; do not push, amend, rebase, or
  rewrite history without separate authorization.
- Never add an AI co-author trailer.
