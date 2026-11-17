# The holdout set

See [the runbook](holdout-runbook.md) for the step-by-step guide to running, auditing and
growing it.

## What it is, and why

`backend/data/gold.yaml` (46 rows) has been read by every session that tuned retrieval or
generation. A number measured on it can no longer tell a real improvement from fitting to the set
(build plan A8). The **holdout** is a second set of real questions, in the same `GoldRow` schema,
that no repository session has seen. It is kept encrypted, outside this repository, and scored by
`eval.holdout`, which never writes the plaintext anywhere and prints only row positions.

## Who runs it

The owner only, in their own terminal, at milestones: the end of a phase, a model change, before a
release. Never inside a Claude session. There is no key in CI, by design.

The private key is protected by a passphrase. `openssl` asks for it on the terminal; it is never an
argument, an environment variable or something Python reads. With no terminal, `openssl` stops at
once and the runner prints "could not decrypt" and exits 2.

## The command

From `backend/`, with the same Ollama, embedder and database settings as an `eval.compare` run:

```bash
EMBEDDER=bge-m3 LLM_PROVIDER=ollama \
  python -m eval.holdout --encrypted /path/to/holdout.p7m --key /path/to/key.pem \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply
```

The database is only read. Run it when nothing else is using Ollama (one LLM job at a time).

The output starts with two lines:

- `holdout sha256:` the first 12 hex digits of the encrypted file's SHA-256, so two reports can be
  tied to the same version of the file;
- `rows:` the number of rows per category.

Then comes the same report `eval.compare` prints. Every row appears as `h01`, `h02`, ... in file
order; the ids the builder chose are never shown. The exit code is `0` when every measured
threshold passes, `1` when one fails, and `2` when the run could not produce a report.

### Looking at what failed

```bash
python -m eval.holdout --encrypted ... --key ... --show
```

After the report, `--show` prints each failing row's `hNN`, category, the checks it failed, its
question and its reference answer. It refuses to run (exit 2) unless stdin and stdout are both
terminals, so it cannot run inside an agent's shell. It does not print the model's answer; put the
question to `/ask` yourself if you want to see it.

## What may be shared

The report: it contains only `hNN`, categories, counts and numbers. Paste it anywhere.

Nothing from `--show` may be shared with a Claude session.

## The burned rule

A question that reaches a Claude session working on this repository is **burned**: the session
could tune against it. Move its row to `gold.yaml`, replace it in the holdout with a new real
question, and re-encrypt the set. This applies to anything that quotes or paraphrases the
question, including a `--show` listing pasted into a chat.

Re-encrypting by hand, in your own terminal (see [the runbook](holdout-runbook.md) §5 for the
full audit workflow this is a part of):

```bash
openssl smime -decrypt -binary -inform DER -in holdout.p7m -inkey key.pem -out holdout.yaml
# edit holdout.yaml, then check that it still loads (from backend/):
python -c "from pathlib import Path; from eval.gold import load_gold_set; load_gold_set(Path('holdout.yaml'))"
openssl smime -encrypt -aes256 -binary -outform DER -in holdout.yaml -out holdout.p7m cert.pem
shred -u holdout.yaml 2>/dev/null || rm -P holdout.yaml
```

The plaintext sits on disk while you edit it, so do this on your own machine and delete it after.

## Guarantees and what keeps them

Nothing the run emits contains a question, a reference answer, a unit id, an original row id or a
model reply: not on stdout, not on stderr, not in a log line.

- Rows are renamed to `hNN` before anything scores them.
- A failure to load prints the exception's type name and a fixed sentence, never its message: YAML
  errors quote the offending line, and row-validation errors quote the row.
- Only `eval.compare`'s own per-row log lines are printed. The `app` loggers do log model replies,
  unit ids and exception messages, so none of their records are printed.
- An exception raised while scoring a row is shown as its type name, never its traceback.
- `tests/test_holdout.py` checks all of this at the file-descriptor level.

Do not set `OLLAMA_DEBUG` on the Ollama server during a run: its log would then record the
prompts, which contain the questions.

When tracing (Langfuse or Sentry) is added to the app, its issue must make this runner switch the
tracing off, and test that: a trace records prompts and replies.
