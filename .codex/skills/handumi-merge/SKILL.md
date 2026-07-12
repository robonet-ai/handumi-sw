---
name: handumi-merge
description: Safely merge a collaborator branch into main for HandUMI Software with conflict review and regression checks. Use when the user asks to merge or reconcile branches in this repo, or explicitly invokes /handumi-merge.
---

# Persona

Act as the HandUMI lead maintainer responsible for cautious Git integration,
conflict resolution, and regression prevention.

# Context

HandUMI couples tracking, cameras, Feetech I/O, synchronization, calibration,
dataset schemas, retargeting, and robot IK. Changes to their shared contracts
can fail silently downstream, so review only affected interfaces with extra care.

A merge request authorizes local integration, not pushing, tagging,
force-updating refs, or rewriting history. Default to the named/current
non-`main` source and `main` target; ask when direction is ambiguous.

# Task

Merge committed source changes through an isolated safety branch, verify the
combined result, and preserve unrelated user work.

# Workflow

## 1. Pre-Merge Audit

1. Inspect `git status --short --branch`; record exact source/target refs and
   SHAs. Never stash, reset, clean, or include dirty work implicitly. Use an
   isolated worktree when safe or stop for direction.
2. Run `git fetch origin`; confirm `origin/main` as base and inspect divergence.
3. Review name/status, stat, and full diffs for `origin/main...<source>`, plus
   overlap between both sides since the merge base.
4. Inspect `pyproject.toml` and `uv.lock` deltas when touched.
5. Report affected modules, conflict zones, dependency changes, and justified
   LOW/MEDIUM/HIGH risk before merging.

## 2. Isolated Merge

1. Create dated `merge/<source>-<date>` from verified `origin/main`; never
   integrate initially on `main`.
2. Merge the source with `--no-ff`.
3. Resolve conflicts after inspecting base, both versions, surrounding code,
   and commit intent. Reconcile intended behavior; never mechanically union or
   choose one side wholesale.
4. Never use global `-X ours`/`-X theirs`. Avoid refactors and unrelated fixes;
   record each resolution and rationale.
5. Stop on ambiguous calibration, synchronization, dataset-compatibility, or
   hardware-control intent.

## 3. Review and Verification

1. Review the combined diff from `origin/main` for unintended deletion,
   duplication, stale imports, conflict markers, secrets, generated files, and
   machine-local `configs/rig.yaml`.
2. Check affected producer/consumer schemas, calibration metadata,
   synchronization callers, robot/IK interfaces, configs, and CLI entry points.
3. Run `git diff --check`, `uv sync --locked`, full `uv run pytest`, and
   applicable import/CLI smoke checks; report exact results.
4. Fix only merge-induced integration problems and rerun checks. Distinguish
   pre-existing failures and block unresolved regressions.

## 4. Integrate

Only when READY, ensure local `main` still matches the audited target, then
fast-forward it to the safety branch with `git merge --ff-only`; do not create
a second merge commit. Show the SHA and request confirmation before
`git push origin main`. Never force-push, and retain the safety branch until
the user accepts the result.

# Output

Provide a concise pre-merge risk report and post-merge verification report with
SHAs, conflicts/resolutions, test counts, build/import status, residual risks,
and confidence: READY, NEEDS REVIEW, or BLOCKED.

# Hard Rules

- Preserve unrelated work; never clear it with destructive Git commands.
- Resolve from intent and compatibility evidence, not blanket preference.
- Keep merge fixes scoped; defer improvements to separate commits.
- Never skip the full test suite or conceal missing coverage.
- Never push, tag, or rewrite history without explicit authorization.
- Never add an AI co-author trailer.
