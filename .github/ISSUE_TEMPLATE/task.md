---
name: Task
about: One concern for one branch and one PR, with a measured problem and runnable acceptance criteria
title: "<what is wrong or missing, stated as the observed behaviour>"
labels: []
---

## What this is

<The problem as measured. Paste the real command and its output that show it, and name the commit it
was measured on.>

## Scope

1. <file or module: what changes>
2. <tests that prove it>

## Out of scope

- <explicit exclusions, including the tempting fix next door>

## Traps

- <the ways this issue has already gone wrong, or is likely to: a probe that regressed another row,
  a fixture that wipes a database, a config that differs between eval and production>

## Blocking

<The results the PR must show before it can merge. Name the database, the number of runs and the
exact rows or questions. Anything listed only as "reported" informs the review but does not block.>

## Acceptance criteria

- [ ] <runnable command, and what its real output must show>
- [ ] `ruff format --check . && ruff check . && mypy app ingest && pytest -q`
- [ ] CI green: paste `gh pr checks <PR>`
- [ ] Dev DB counts before and after, if the dev database is used

Branch: `<feat|fix|chore|docs>-<slug>`. Start from `main` at or after `<sha>`.
