# Collaboration Workflow

This repository is built by Claude sessions with fixed roles, coordinated by the human owner. Each
issue is implemented by a fresh session that starts with no context: this file, the issue and the
hand-off prompt are all it reads, so every rule a session must follow lives in one of those three.

## Roles

| Role | Who | Responsibility |
| --- | --- | --- |
| **Planner / Reviewer** | A long-running Claude session | Writes issues and hand-off prompts. Reviews each PR by reproducing the claims that matter, not only by reading pasted output. Merges a PR that verifies clean, then re-tests the product on `main`. Does not write feature code. |
| **Implementer** | A new Claude session per issue | Works in its own worktree. Implements exactly the issue's scope, runs every acceptance command, opens a PR, reports to the owner, then stops. **Never merges.** |
| **Owner** | Human | Relays hand-off prompts and reports. Decides design questions and every exception to a gate. Manages secrets. Runs the commands the sessions are not allowed to run, such as dropping a database or force-removing a worktree. |

## Per-issue flow

1. **Plan (planner).**
   - Open an issue from the `Task` template (`.github/ISSUE_TEMPLATE/task.md`): the measured problem,
     scope, out of scope, traps, blocking criteria, acceptance criteria and the branch name.
   - Write a hand-off prompt from the template below. The owner gives it to a new session.
2. **Implement (implementer).**
   - Create a worktree and branch from the latest `origin/main` (see **Worktrees**).
   - Build only the listed scope. If something is missing, comment on the issue first and wait until
     it is added to scope.
   - Run every acceptance command, push, and open a PR from the PR template with `Closes #N`.
   - Report to the owner with the implementer report frame below, then stop.
3. **Review (planner).**
   - Read the diff and the pasted output, and check each acceptance criterion against the code.
   - Reproduce independently every claim the decision rests on: re-run the measurement, ask the real
     `/ask` server, read the real corpus. Pasted output alone is not verification.
   - Change requests are PR comments with file, line, what to change and why. The implementer pushes
     fixes to the same branch and replies to each comment with the commit SHA that addresses it.
4. **Merge (planner).** Since 2026-09-17 the planner may merge a PR when all of these hold:
   - CI is green on the PR's head commit (`gh pr checks <PR>`);
   - no change request is open;
   - the independent verification found nothing.

   Merge with `gh pr merge <PR> --squash --match-head-commit <sha>`.
5. **Verify on `main` (planner).** After every merge:
   - CI is green on the merge commit;
   - the real product answers as before on the dev database, through `/ask`;
   - the dev database counts are unchanged.
6. **Exceptions (owner).**
   - When a gate fails, the owner decides whether the PR merges, and the planner merges only on an
     explicit "merge N". A gate is a threshold in the build plan, a red CI check, or a blocking
     criterion in the issue.
   - The planner gives the owner a recommendation and its reason.
   - The planner records the exception and the reason as a comment on the PR.

## Worktrees

Two sessions must never share a working directory: one session's `git stash` or `git checkout` pulls
the floor out from under the other. Each implementation gets its own worktree, inside the repository:

```bash
cd /Users/rhynis/Projects/gascomply
git fetch origin
git worktree add .worktrees/<short-name> -b <branch> origin/main
cd .worktrees/<short-name>/backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`.worktrees/` is git-ignored. Never run `git stash`, `git checkout` or `git add -A` in the main checkout
at `/Users/rhynis/Projects/gascomply`. Remove your worktree once the PR is merged, because each one
carries its own virtualenv and costs a few hundred megabytes:

```bash
git worktree remove .worktrees/<short-name>
```

## Databases

- **Dev database `gascomply`.** Implementers read it and never write to it. No re-ingest unless the
  issue says so. Paste its counts before and after any run that uses it:

  ```bash
  docker exec gascomply-postgres-1 psql -U postgres -d gascomply -Atc "select (select count(*) from documents)||' documents, '||(select count(*) from units)||' units, '||(select count(*) from chunks)||' chunks, '||(select count(*) from edges)||' edges, '||(select count(*) from edges where review_status='approved')||' approved, '||(select count(*) from edges where review_status='pending')||' pending, '||(select count(*) from edges where review_status='rejected')||' rejected'"
  ```

- **Tests.** The test fixture truncates every table of the database it is given. Tests use
  `TEST_DATABASE_URL`, which must name a database ending in `_test`. Create one per issue and drop it
  when done:

  ```bash
  docker exec gascomply-postgres-1 createdb -U postgres gascomply_<N>_test
  export TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply_<N>_test"
  ```

  Never point `pytest` at the dev database.

## LLM jobs

- **One LLM-generating job at a time** on this machine: one `eval.compare` run, or one API server
  serving `/ask`. Two at once ran the machine out of memory and restarted it on 2026-10-31. Check
  first, and stop each server when done:

  ```bash
  pgrep -fl "^[^ ]*[Pp]ython[^ ]* .*(eval\.(compare|holdout)|uvicorn|-m app( |$))" && echo "busy — wait" || echo "clear to run"
  ```

  The pattern is anchored to the process's own executable so that another session's waiting loop,
  which quotes this text, does not count as a job.
- Start a local `/ask` server as `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`, so
  the check names it plainly; stop it when done.
- `qwen3:8b` is the default model. Do not pull or remove models, and do not set `OLLAMA_MODEL` unless the
  issue says so. If Ollama returns HTTP 500, restart it:
  `pkill -f "ollama serve"; nohup ollama serve &`.
- **Measuring a change to retrieval or generation:**
  - Run the `main` baseline first, from the main checkout's code and its own `.venv`. Then run the
    branch from the worktree, in the same session.
  - The model's output varies from run to run. Run the branch at least twice, and record each run's
    wall time.
  - Explain every row that moves from that row's own run. "Flicker" alone is not an explanation.
  - A move on a row the change should not reach means the change leaked. Print that row's pool.
- **Eval generates the way `/ask` does.** Retrieval width, prompt and model are the same as `/ask`'s,
  so the eval measures the product rather than a variant of it.

## Ground rules

- **One concern, one branch, one PR.** Never mix issues.
- **No silent scope creep.** Comment on the issue first, get it added to scope, then implement.
- **Real commands, real output.** Never paste expected output. Write "not measured" for anything
  you did not measure.
- **Sources of truth.** The issue and its comments define the current task, and a newer comment wins
  where they differ. The issue wins over the build plan.
- **Language (C8).**
  - English: code, comments, docstrings, commit messages, PR text, issues and hand-off prompts.
  - Vietnamese: the product's UI strings, the product's own LLM prompts and replies, the legal corpus,
    the gold questions, and reports to the owner.
- Docstrings say what the code does. The history of attempts that were discarded goes in the PR body.
- Pre-existing failures are called out in the PR body, with evidence that `main` fails the same way.

## Hand-off prompt (English)

```
Repo: https://github.com/Rhynis/gascomply
Working directory: /Users/rhynis/Projects/gascomply/.worktrees/<short-name>/backend

Read in this order:
1. /Users/rhynis/Projects/gascomply/WORKFLOW.md
2. GitHub issue #N and every comment on it: https://github.com/Rhynis/gascomply/issues/N
3. <any PR or file the issue builds on>

Rule: the issue and its comments are the source of truth (newer comments win where they differ).
Create branch `<branch>` from main (`git log origin/main --oneline -1` shows <sha> or later),
implement, run every acceptance command, and open a PR titled
"[<Area>] <title>"
whose body contains `Closes #N` plus verbatim terminal output.

## Worktree setup        <the commands from "Worktrees" and "Databases", filled in>
## Running in parallel   <what else is running now; the one-LLM-job rule if the issue uses the LLM>
## Database              <which database each command uses; the counts that must not change>
## Traps                 <the ways this issue has already gone wrong, or is likely to>
## Verify before pushing
ruff format --check . && ruff check . && mypy app ingest && pytest -q
## Rules                 <C8; real output; "not measured">
## When done             Push, open the PR, report to the owner in Vietnamese using exactly the
                         implementer report frame, then STOP. Never merge.
```

- **Delivery.** The planner saves the hand-off as
  `/Users/rhynis/Projects/gascomply-archive/handoffs/handoff-<N>.txt`. The owner starts the
  implementer with one line: `Read <that path> and follow it exactly.`
- **Fix round** (changes requested on an open PR): same header and sections, but the "Create
  branch" sentence becomes *"This is a fix round on the existing branch `<branch>` (PR #M, head
  `<sha>`). Do not open a new PR. Merge `origin/main` first, implement the review comment `<url>`,
  re-run every acceptance command, push, and update the PR body."* The report's first line becomes
  "Đã xong vòng sửa issue #N." The file is `handoff-<N>-fix.txt`.

## Implementer report (Vietnamese, plain text)

```
Đã xong issue #N.

PR: <url> (OPEN)
Branch: <tên branch>
Commit: <hash> <tiêu đề commit>

Đã implement:
<file:điểm chính, mỗi dòng một bullet>

Acceptance:
<từng command đã chạy thật>
Kết quả: <số liệu/output thật>

Worktree note: <môi trường, DB đã dùng, model còn cài, lưu ý>
```

## Reviewer report (Vietnamese, plain text)

```
Đã review PR #N (issue #M): <đã merge | chưa merge, vì …>

Merge: <hash> trên main, CI trên merge commit: <xanh | đỏ, lý do>

Tự đo lại:
<từng phép đo reviewer tự chạy, kết quả thật>

Test sản phẩm trên main:
<từng câu hỏi qua /ask và kết quả>
Dev DB: <counts>

Còn mở:
<lỗi đã biết, việc chờ owner quyết hoặc chạy>

Tiếp theo: <issue kế tiếp, hand-off nằm ở đâu>
```

## Communication artifacts

| Artifact | Purpose |
| --- | --- |
| GitHub issue | Task definition: measured problem, scope, traps, acceptance criteria |
| GitHub PR | Implementation, verbatim acceptance output, review thread, recorded exceptions |
| Hand-off prompt | Per-issue environment and traps for a new implementer session |
| `BUILD_PLAN.md` | Long-term reference. A git-ignored symlink into the owner's private `build-foundation` repository (`projects/gascomply/BUILD_PLAN.md`) |
| `BUILD_PLAN_FOUNDATION.md` | Reusable, project-agnostic template for build plans, workflow, code-quality standards, issue/prompt/report templates, CI/CD, lessons learned, repo bootstrap and operations. A git-ignored symlink into the owner's private `build-foundation` repository |
| This file | Process documentation |
