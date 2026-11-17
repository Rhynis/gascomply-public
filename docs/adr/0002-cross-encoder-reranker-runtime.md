# ADR 0002: Rerank with a cross-encoder served by a `llama-server` sidecar

## Status

Accepted (build plan A6 Phase 3.4's "optional bge-reranker", issue #114).

## Context

Retrieval already finds the right text: recall@k is 0.900 against a 0.800 threshold. What fails
is the generator's model choosing a *different Điều* out of too many passages -- issue #108 found 9
of 12 wrong citations were exactly that, and after #107 and #111 a single question offers up to
22 labelled `[unit_id: ...]` lines (7944 prompt characters for "mở cửa hàng bán lẻ gas LPG").

A vector search scores a passage by the distance between two *independent* embeddings, the
question's and the passage's. A **cross-encoder** reads the (question, passage) pair *together* and
outputs one relevance score for the pair, which is the difference between "the article about this
exact question" and "an article about the same subject". Scoring every retrieved passage that way
and keeping only the best few removes the wrong options before the model sees them.

The runtime is the decision. **Ollama has no rerank endpoint** -- an open request since March 2024
with several stalled PRs (ollama#17129). Community `bge-reranker-v2-m3` uploads exist on the Ollama
registry, but served through `/api/embeddings` they expose only the embedding layer, not the
classification head. Scores taken from there would look like scores and not be cross-encoder scores
-- a measurement that looks real and isn't, which this project has refused to ship. **No reranker
score in this repository comes from `/api/embeddings`.**

## Decision

**Run the cross-encoder as a sidecar: `llama-server --rerank` (llama.cpp) serving the
`bge-reranker-v2-m3` GGUF.** `app/rerankers/llama_server.py` calls its `POST /v1/rerank`.

That the scores are a real cross-encoder's is checked, not assumed. The GGUF carries the
classification head an embedding endpoint never exposes -- read from the file with the `gguf`
package:

```
cls.weight        [1024, 1024]  Q8_0      cls.bias         [1024]  F32
cls.output.weight [1024]        F32       cls.output.bias  [1]     F32
```

(a dense layer, then a projection to a single logit), and the server log line
`model default pooling_type is [-1], but [4] was specified` is `--rerank` selecting `rank` pooling,
which is that head. `relevance_score` is the raw logit: unbounded, negative for an irrelevant
passage, comparable only within one request. Sanity check with a real sidecar (question: minimum
area of an LPG retail shop; four passages):

| passage | score |
|---|---|
| Điều 53: "Cửa hàng phải có diện tích tối thiểu là 12 m2 ..." | 0.157 |
| cylinder inspection | -4.885 |
| licence for general LPG agents | -7.021 |
| "Hôm nay trời đẹp, tôi đi chợ mua rau và cá." | -11.018 |

### What was run

| | |
|---|---|
| binary | llama.cpp release **b11178** (`llama-b11178-bin-macos-arm64`), `llama-server --version`: `0.5.0-dev (build 11178, commit f9af9be21)` |
| model | `gpustack/bge-reranker-v2-m3-GGUF`, `bge-reranker-v2-m3-Q8_0.gguf`, 635,676,416 bytes, sha256 `a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3` (matches Hugging Face's LFS digest), Apache-2.0 |
| command | `llama-server -m bge-reranker-v2-m3-Q8_0.gguf --rerank --port 8081 -c 8192 -b 8192 -ub 8192 -np 1` |

`-ub` must be at least the longest passage plus the question: a non-causal model processes its whole
input in one physical batch, so a smaller `-ub` is an error, not a slowdown. The longest chunk in
the corpus is 5,574 characters; 8192 tokens covers it many times over. The server binds to
127.0.0.1 by default and is not meant to be exposed.

### Configuration

| setting | values | meaning |
|---|---|---|
| `RERANKER` | `none` (default), `mock`, `llama-server` | which reranker `Retriever` uses |
| `RERANKER_BASE_URL` | default `http://localhost:8081` | the sidecar. Never Ollama's URL |
| `RERANK_TOP_N` | default 5 | passages kept after reranking (chosen below) |

`none` is the default, unlike `EMBEDDER=mock` and `LLM_PROVIDER=mock`. Those stand in for stages
nothing works without; reranking is optional, so its offline default is to skip it -- exactly the
behaviour before this issue -- rather than a mock that would quietly cut real passages by word
overlap in a deployment that never asked for it. `mock` (`MockReranker`: the share of the question's
words a passage contains) exists so the whole path -- scoring, reordering, the cut, the fallback --
is exercised in CI with nothing installed, the way `MockEmbedder` and `MockLLM` do it.

### Where it sits

`Retriever.retrieve` ranks by graph proximity as before, takes the `passage_limit` survivors, scores
them with the cross-encoder, reorders best first and keeps the top `RERANK_TOP_N`. Focus expansion
(#106) then runs on the survivors only, so every clause of a dropped passage stops being offered as
citable -- most of the prompt. Widening the pool was tried and bought nothing: with `passage_limit`
raised to 30 the candidates number 13.5 per question on average, and recall was identical to the
10-passage pool (0.904 either way), so the cross-encoder reorders what the graph chose and does not
fetch more. Entities are not reranked (out of scope), and neither is the naive path.

### Failure mode

If the sidecar is down, times out (60 s), answers non-2xx, or answers with anything other than
exactly one finite score per document, `LlamaServerReranker` raises `RerankError`. `Retriever` logs a
warning and keeps the graph-proximity order, so `/ask` answers exactly as it did before reranking
existed. Only `RerankError` is caught: any other exception is a bug in a reranker, and hiding it
behind a fallback would be a silent measurement of the wrong thing. For the same reason
`python -m eval.compare` counts fallbacks and prints them at the top of the table, with a warning if
there were any -- the one place where silent degradation must not be silent.

## Alternatives considered

**FlagEmbedding / sentence-transformers, in process.** Works, and needs no second process. Rejected:
it pulls in `torch`. On the platform CI runs (Linux x86_64, CPython 3.11) the `torch 2.14.0` wheel
alone is 555 MB on PyPI, before its `nvidia-cudnn`/`nvshmem`/`nccl`/`cusparselt` dependencies
and `transformers`, `scikit-learn` and `scipy` from `sentence-transformers 6.1.0`; on macOS arm64
it is 127 MB. That is a very large addition to a backend whose whole dependency list is a dozen
small packages (C12), in every install and every CI run, for a stage that is optional and off by
default. A model that big is also loaded into the API process itself, next to Postgres, Ollama and
`bge-m3` on a 16 GB machine.

**Listwise reranking with the existing chat model.** No new dependency. Rejected on the evidence of
what it would be: it asks the *same* `qwen2.5:7b-instruct-q4_K_M` whose passage-picking weakness is
the problem to pick among the same passages -- the failure moved one step earlier, not removed -- and
adds a model call to every question on a path where one call already takes tens of seconds.

**Reuse the `llama-server` bundled inside Ollama.** Ollama 0.30 ships one
(`.../libexec/lib/ollama/llama-server`, build 9672) and it does accept `--rerank`. Rejected as
something to depend on: it is an implementation detail of Ollama's runner -- its path, its flags and
its build change with Ollama's releases -- not an interface anyone promises to keep.

**Serve the reranker through Ollama's `/api/embeddings`.** Rejected outright, above.

## Consequences

- **One more process** to run when reranking is on. Off by default; when it is on but down, `/ask`
  degrades instead of failing.
- **Memory, measured** on the 16 GB development machine (Apple M4), `llama-server` b11178 with the
  Q8_0 model and the command above:

  | | |
  |---|---|
  | physical footprint (`footprint -p`), idle after load | 1414 MB (591 MB of it the memory-mapped model file, the rest dirty allocations) |
  | physical footprint after scoring the longest passage in the corpus | 1630 MB |
  | resident set (`ps` RSS), idle | about 620,000 KB |
  | highest RSS sampled every 5 s across the five reranked full evaluations | 663 MB |

  It is a real addition to a machine already running Ollama, `bge-m3` and Postgres: during these
  runs the Ollama chat-model runner (`qwen2.5:7b-instruct-q4_K_M`, 4096-token context) had a
  footprint of 13 GB by the same tool (about 4.5 GB RSS), and the system reported about 20% memory free.
  Shrinking the sidecar's context does not help: with `-c 4096 -b 4096 -ub 4096` and with
  `-c 2048 -b 2048 -ub 2048` the footprint was 1673 MB, not smaller, and the scores for the same
  three passages (the longest chunk, 1,493 tokens, among them) were identical to five decimal
  places to those from `-c 8192`. So the command keeps 8192, the model's own maximum, which cannot
  reject a long article a future corpus adds.
- **Latency.** Scoring the ten passages of one question took a mean of 3.0 s (min 0.9 s, max 6.1 s)
  over the 26 answerable gold questions (mean 10,081 characters per question), with nothing else
  running on the machine; scoring thirty passages took a mean of 3.4 s. That is added to every
  `/ask` that reranks, and it is partly paid back: the model reads fewer passages. From the
  timestamps in the evaluation logs (time from the previous logged request to the model's reply,
  hybrid path, about 30 replies per run) the mean generation time was 27.2 s and 27.4 s without
  reranking, and 24.2 s at N=5 (two runs: 24.8 s and 23.6 s), 16.0 s at N=3, 12.4 s at N=2 -- so at
  N=5 the saving roughly cancels the scoring cost, and at N=2 and 3 it more than covers it. These
  are derived from log timestamps, not from an instrumented benchmark, and the model's own
  run-to-run variation (issue #109) is in them.
- **CI is unaffected.** `MockReranker` and `httpx2.MockTransport` cover every path; no test needs the
  sidecar, none is skipped, and `tests/test_rerankers.py` also drives the client over a real local
  socket against a recorded `llama-server` answer.

## Choosing N by measurement

Fewer passages can cut recall: drop the passage holding the gold clause and recall@k falls, and it is
the one threshold with a comfortable margin (0.900 against 0.800). So N is chosen by measuring recall@k
and citation accuracy together, at every N tried, not by taste.

Everything below is from the real corpus (73 documents, 2311 units, 706 chunks), real `bge-m3`
embeddings, real `qwen2.5:7b-instruct-q4_K_M` through Ollama, and the real sidecar. The 38-row gold
set has 26 answerable rows. Reranking touches the hybrid path only; the naive path is never reranked.

### Retrieval only: what N does to recall and to the prompt

No model call, all 26 answerable rows, every N, through `Retriever.retrieve` itself with the real
sidecar. Recall is computed exactly as `eval.compare` computes it. "Lines" is the number of
`[unit_id: ...]` lines the model is shown, the count issue #108 measured. The `eval` configuration is
what `python -m eval.compare` runs (`k=10`, `passage_limit=10`); the `deployed` one is what `/ask`
runs (`k=5`, `passage_limit=15`, `GENERATION_RETRIEVAL_CONFIG`), which recalls less to begin with.

| top-N | recall@k eval | lines eval | prompt chars eval | recall@k deployed | lines deployed | prompt chars deployed |
|---|---|---|---|---|---|---|
| off | 0.904 | 56.9 | 17,586 | 0.827 | 47.1 | 15,208 |
| 1 | 0.788 | 6.3 | 2,666 | 0.712 | 6.5 | 2,606 |
| 2 | 0.865 | 11.7 | 4,420 | **0.788** | 12.0 | 4,255 |
| 3 | 0.885 | 17.7 | 6,116 | 0.808 | 17.2 | 5,810 |
| 4 | 0.885 | 22.9 | 7,795 | 0.808 | 22.5 | 7,381 |
| 5 | 0.885 | 28.7 | 9,451 | 0.827 | 27.8 | 8,994 |
| 6 | 0.904 | 33.7 | 11,019 | 0.827 | 32.6 | 10,518 |
| 7 | 0.904 | 40.6 | 12,844 | 0.827 | 38.7 | 12,299 |
| 8 | 0.904 | 46.0 | 14,432 | 0.827 | 42.3 | 13,391 |
| 10 | 0.904 | 56.9 | 17,586 | 0.827 | 45.6 | 14,507 |

Two things fall out. **N=1 is out** (0.788 and 0.712, both under 0.800), and **N=2 is out for the
deployed configuration** (0.788), though it passes in the evaluation one. And the cost of the cut is
one gold clause: `multihop-tank-condition-amended`'s `dieu-15#khoan-2` sits at cross-encoder rank 6,
so any N below 6 loses it in the evaluation configuration (0.904 to 0.885). At N=5 the deployed
configuration loses nothing (0.827 either way) and the prompt is 41-46% shorter.

Widening the pool instead (`passage_limit` 30, then keep N) recalled exactly what the 10-passage pool
did, for every N: the graph ranking already puts everything findable in its top ten.

### Full `python -m eval.compare`, every N tried

The real table each run printed, first lines and all: `Reranker: llama-server, top-N 5 -- ... 33
retrieval(s) reranked, 0 fell back`. Every reranked run below reranked every retrieval and fell back on
none. Recall and citation accuracy are means over the rows that run *scored*: a row whose model call
times out twice (issue #109) is excluded from every metric, and which rows those are changes from run to
run, so the last column names them and `n` counts what was left. All runs are the evaluation
configuration, `RERANKER` and `RERANK_TOP_N` set in the environment, nothing else changed, run one at a
time with the machine otherwise idle of evaluations.

| run | top-N | n | recall@k (strict) | recall (containment) | citation accuracy | effective-answer rate | rows not scored |
|---|---|---|---|---|---|---|---|
| baseline a | off | 25 | 0.900 | 0.900 | 0.429 | 0.840 | old-pccc-law-repealed |
| baseline b | off | 25 | 0.900 | 0.900 | 0.435 | 0.920 | old-pccc-law-repealed |
| n2 | 2 | 26 | 0.865 | 0.865 | 0.627 | 0.962 | none |
| n3 | 3 | 25 | 0.880 | 0.880 | 0.587 | 0.920 | min-store-area |
| n5 | 5 | 24 | 0.875 | 0.875 | 0.611 | 0.875 | min-store-area, old-pccc-law-repealed |
| n5b | 5 | 24 | 0.875 | 0.875 | 0.532 | 0.875 | min-store-area, display-and-inspection |
| n8 | 8 | 25 | 0.900 | 0.900 | 0.458 | 0.960 | old-pccc-law-repealed |

Every one meets the 0.800 recall threshold. None meets the 0.850 citation-accuracy threshold, and
reranking does not claim to: that gap is the generator's (#113, #112).

To compare on the same rows, this is each reranked run against baseline b on the rows *answered* in
both (a row the model refused, or a run did not score, is left out of that pair):

| run | top-N | rows in both | baseline b | reranked | rows up | rows down | rows unchanged |
|---|---|---|---|---|---|---|---|
| n2 | 2 | 22 | 0.455 | 0.599 | 5 | 2 | 15 |
| n3 | 3 | 20 | 0.450 | 0.625 | 5 | 1 | 14 |
| n5 | 5 | 19 | 0.421 | 0.588 | 4 | 1 | 14 |
| n5b | 5 | 18 | 0.389 | 0.528 | 3 | 0 | 15 |
| n8 | 8 | 22 | 0.455 | 0.455 | 0 | 0 | 22 |

### How much of that is noise

The generator is a local model at temperature 0.1, and there are about twenty answered rows per run,
so one row is worth about 0.05. Two pairs of identical configurations were run:

- **baseline a vs b** (no reranking): citation accuracy 0.429 vs 0.435 (0.006 apart), recall 0.900 both,
  but the effective-answer rate 0.840 vs 0.920 (0.08 apart).
- **n5 vs n5b** (identical, N=5): citation accuracy 0.611 vs 0.532 (0.079 apart), recall 0.875 both. On
  the rows both scored, one row changed (`compliance-blending-lab`, 1.0 to 0.0); the rest of the gap is
  which rows each run managed to score.

So movement of up to about 0.08 between identical runs is normal here, and **differences among N=2, 3
and 5 (0.587 to 0.627, and 0.532 to 0.611 for the two N=5 runs) are inside it: these runs do not say
which of the three is best.** What they do show is separation from the runs that did not cut: every
run at N of 5 or fewer (four runs) scored above the three runs that did not cut meaningfully, the two
baselines and N=8 (0.532 to 0.627 against 0.429 to 0.458), and on shared rows reranking moved 17 rows up and 4 down across those four runs, gaining
`multihop-pccc-law-and-decree`, `multihop-decree-87-amended` and `compliance-blending-registration` in
both N=5 runs. The lowest run at N of 5 or fewer (0.532) is 0.074 above the highest of the other three, about one
noise-width, so the size of the gain is uncertain (roughly +0.1 to +0.2) while its direction is
consistent. At **N=8 nothing changed at all**: on the 22 rows answered in both runs, every row's
citation precision was identical to baseline b's. Reordering and trimming two passages does not help;
what helps is showing the model few enough options.

### Decision: `RERANK_TOP_N` = 5

The constraint is recall: at least 0.800 in the evaluation *and* in the deployed configuration, where
recall starts at 0.827 and has little room. N=1 and N=2 fail it. N=3 passes the deployed one with a
margin of 0.008, and loses a gold clause there for no citation gain the noise lets us see. **N=5 is the
smallest N that costs the deployed configuration no recall at all (0.827, unchanged), and it sits on
the side of the gain (0.611 and 0.532 against 0.429 and 0.435 unreranked).** Its evaluation-configuration
recall is 0.885 against 0.904, one gold clause (`dieu-15#khoan-2`), for a prompt 41-46% shorter.

A smaller N would probably score a little higher on citation accuracy (N=2 scored 0.627 and had the
fewest timeouts), but it would do so by dropping gold clauses in the configuration that ships, and this
project treats a missing citation as worse than a wrong one it can still catch. Revisit N if the
generator changes (#113, #112) or the corpus grows: both move where the gold passage ranks.

### What this does not show

- Fifty-odd model calls per run at temperature 0.1: the ranking among N=2, 3 and 5 is not established,
  only that N of 5 or fewer beats no cut on these rows.
- The gold set is 26 answerable questions from one corpus, and each run lost some of them to model
  timeouts (issue #109). `compliance-min-store-area` (the flagship) timed out on its first attempt in
  all five reranked runs and in neither baseline run, and twice in a row in three of the five. That
  looks like an effect of reranking and is not: in the six runs made with the final harness, 18 of the
  19 timeouts were in the *naive* path's model call (`naive_response`, ten whole articles that overflow
  the model's 4096-token window and are never reranked), and only one in the hybrid path (n8). Run in
  isolation, the flagship's own hybrid prompt never timed out: 27.2 s and 23.1 s unreranked, 21.9 s
  and 22.6 s at N=5, 12.1 s and 3.4 s at N=2 (two replies each, prompts of 22,277, 10,154 and 3,141
  characters). The rows a run loses are the harness's coin flip on the naive side, which is why the
  comparison above is also given on the rows both runs answered.
- Entities are not reranked (out of scope), so a requirement whose source passage was cut still lists
  that source's `unit_id` for the model to cite.
