# How many holdout rows: the math, with this project's numbers

This explains how the holdout's target size was worked out, so that it can be re-checked or re-done later. It assumes no statistics background. Every example uses numbers this project has measured.

The holdout's own workflow is in [the runbook](holdout-runbook.md); this page only explains §6.1 of it.

## 1. Why the size matters: one example

The first holdout run (2026-11-06) scored citation accuracy **0.500**. That is 7 correct out of the 14 answerable rows the system answered.

With only 14 rows, chance alone moves that number a lot. A different set of 14 equally fair questions could easily have scored 5/14 = 0.36 or 9/14 = 0.64. Section 2 shows that the true value is anywhere from about **0.27 to 0.73**. One run on 14 rows cannot say whether citation is "about half right" or "about three-quarters right". More rows make the range narrower.

## 2. How sure one number is: the margin of error

A score like citation accuracy is a proportion `p` (correct ÷ total) measured on `n` rows. Its **95% margin of error** is

    E = 1.96 × sqrt( p × (1 − p) / n )

and the 95% **confidence interval** is `p − E` to `p + E`.

**What "95%" means.** Suppose fresh sets of n questions, drawn the same way, were scored again and again, and each run's interval computed like this. About 95 runs in 100 would give an interval that contains the system's true score. The number 1.96 is what makes it 95%: in a normal (bell-curve) distribution, 95% of values fall within 1.96 standard deviations of the middle. A 99% interval would use 2.58 and be wider. A 90% interval would use 1.645 and be narrower.

**Example: the holdout run.** p = 0.500, n = 14:

    E = 1.96 × sqrt(0.5 × 0.5 / 14) = 1.96 × 0.134 = 0.26

The interval is 0.24 to 0.76. The simple formula is rough at n = 14. The more accurate Wilson interval (section 7) gives **0.27 to 0.73**.

**Example: the gold set.** Gold citation accuracy was 0.760 on about 32 answered rows:

    E = 1.96 × sqrt(0.76 × 0.24 / 32) = 0.15

So gold's own number is only known to within about ±0.15, from about 0.61 to 0.91.

**Turning it around to size the set.** Solve the formula for n:

    n = 1.96² × p × (1 − p) / E²

`p × (1 − p)` is largest at p = 0.5, so p = 0.5 gives the **worst case**, the size that is enough whatever the score turns out to be:

| Margin you want | Rows needed at p = 0.5 | at p = 0.75 |
|---|---|---|
| ± 0.15 | 43 | 32 |
| ± 0.12 | 67 | 50 |
| **± 0.10** | **96** | **73** |
| ± 0.05 | 385 | 289 |

And the other way round, the margin you get from a given number of rows:

| Rows | Margin at p = 0.5 | at p = 0.75 |
|---|---|---|
| 14 (the first run) | ± 0.26 | ± 0.23 |
| 40 | ± 0.15 | ± 0.13 |
| 70 | ± 0.12 | ± 0.10 |
| 96 | ± 0.10 | ± 0.09 |
| 150 | ± 0.08 | ± 0.07 |

Two things to notice:

- **Precision grows with the square root of n.** Halving the margin takes four times the rows. That is why ±0.05 costs 385 rows while ±0.10 costs 96.
- **96 is the worst-case size for ±0.10.** The system's citation accuracy is somewhere between 0.5 and 0.76, so 70–96 rows give about ±0.10. 96 makes sure of it.

**n counts the rows that are scored, not the rows written.** Citation accuracy is computed only on answerable rows the system *answers*; a refusal is not scored for citation. The measured effective-answer rate is 0.93–0.94. So to have about 96 scored rows, write about **100 answerable rows**.

**Citation scores are not always 0 or 1.** A multi-hop row that cites 1 of its 3 gold clauses scores 0.333. Scores between 0 and 1 vary less than pure 0/1 scores do, so the formula above slightly overstates the margin. It errs on the safe side. In the first holdout run every row happened to score exactly 0 or 1, so the formula is exact there.

## 3. Is a difference real? Significance and power

The holdout is used to answer three kinds of questions. Each needs its own calculation.

### 3.1 Two words: significance and power

When you compare two numbers, two different mistakes are possible:

- **A false alarm:** you conclude there is a difference when there is none and chance made the numbers differ. The **significance level α** is the false-alarm rate you accept. The usual choice is α = 0.05, a 5% chance. It matches the 95% of section 2, and gives the same 1.96.
- **A miss:** a real difference exists, but the sample was too small to show it. The **power** is the chance of *not* missing a real difference of a given size. The usual choice is 80%. Its number is 0.84, the point of the bell curve that leaves 20% below it.

A **p-value** is the chance of seeing a difference at least this large if there were really none. "p < 0.05" means the difference passes the 5% false-alarm bar.

### 3.2 Did the gold number carry over? (two different question sets)

Gold says citation 0.760 (n ≈ 32), and the holdout says 0.500 (n = 14). Is the 0.26 drop real?

The two sets are different questions, so each one's uncertainty adds:

    SE = sqrt( 0.76 × 0.24 / 32  +  0.5 × 0.5 / 14 ) = sqrt(0.0057 + 0.0179) = 0.154
    z  = gap / SE = 0.26 / 0.154 = 1.69

z = 1.69 is below 1.96, so the gap is **not** significant at the 5% level (p ≈ 0.09). The honest reading of the first run: the drop is *likely* real, but 14 rows cannot prove it. The finding that the grounding verifier kept more wrong citations than right ones (8 against 7) points the same way, and the next runs must confirm it.

Take the same gap with 96 holdout rows instead:

    SE = sqrt(0.0057 + 0.5 × 0.5 / 96) = 0.091,   z = 0.26 / 0.091 = 2.85   (p ≈ 0.004)

Now the gap is clearly significant. Part of the remaining uncertainty is gold's own, from its 32 rows, and more holdout rows cannot remove that.

### 3.3 Is version B better than version A? (the same questions, twice)

To compare two systems, such as main and a branch, or qwen3:8b and gemma4, run both on the **same** rows. Most rows score the same both times. Only the rows that **change** carry information. This is called a *paired* comparison, McNemar's test.

If a share `d` of rows change between the two runs, the rows needed to detect a gap Δ with α = 0.05 and 80% power are

    n ≈ (1.96 + 0.84)² × d / Δ²

In the review of PR #188, 6 of 34 answerable gold rows changed between main and the branch, so d ≈ 0.18. The table uses d = 0.20:

| Gap Δ to detect | Rows needed |
|---|---|
| 0.20 | 40 |
| 0.15 | 70 |
| about 0.13 | 96 |
| 0.10 | 157 |
| 0.05 | 627 |

So **a 96-row holdout can reliably detect a change of about 13 points** between two versions. Smaller changes, like a 3–5-point retrieval tweak, are for the gold set and its per-row diffs; the holdout cannot settle them. That is the holdout's job description: it confirms big effects on unseen questions and catches over-fitting to gold. It does not rank small tweaks.

### 3.4 Does it miss emergencies? (when the answer should be "never")

For a rate that should be zero, such as missed emergencies, use the **rule of three**: if 0 of n rows fail, the true failure rate is below about **3 / n**, with 95% confidence.

| Emergency rows, 0 missed | The miss rate is below |
|---|---|
| 4 | 75% |
| 15 | 20% |
| 30 | 10% |
| 100 | 3% |

Proving a miss rate below 1% would take 300 emergency rows with none missed, far more than a holdout can hold. So **the safety guarantee rests on `check_emergency`'s rules and their unit tests**, and on the gold set. The holdout's 15 emergency rows probe phrasings the rules were not written for: no diacritics, no question mark, slang, a typo. Each miss found that way becomes a new test.

## 4. What the size of the knowledge base changes, and what it does not

It is natural to ask whether the holdout should be some share of the knowledge base, for example "10% of the articles". It should not, and the reason is worth understanding.

**The margin of error does not depend on the size of the corpus.** The formula in section 2 has no corpus size in it. What the holdout samples is *the questions a shop owner might ask*, a practically unlimited set, not the 248 articles. A set of 96 questions gives about ±0.10 whether the corpus has 248 articles or 2 480.

There is a *finite population correction* for sampling from a small, fixed population. Treating the 248 articles as the population would cut 96 to about 69. It does not apply here, because one article can be asked about in many different ways.

**The corpus does decide where the rows go.** Here it matters in two ways.

1. **Coverage.** Spread the rows over the documents the system can answer from. The 7 documents with text hold 248 articles: Nghị định 87/2018 62, Luật PCCC 2024 55, Nghị định 105/2025 47, QCVN 10:2012 25, Nghị định 17/2020 23, QCVN 04:2013 22 and Thông tư 36/2025 14. Not every article matters to a shop owner. The split in section 5 follows relevance, not article count.
2. **Clustering.** Two questions about the same article tend to succeed or fail together. They are worth less than two independent questions. Statisticians call this the *design effect*: with m rows per article and a similarity ρ between them, n rows count as only `n / (1 + (m − 1) × ρ)` independent ones. For example, with ρ = 0.3, 96 rows count as 96 at one row per article, 74 at two rows per article, and 60 at three. That is why the runbook **caps each article at 2 rows**, and why one row per article is better where there are enough articles.

## 5. The target for this project

Taken together:

| Stage | Answerable (compliance + multi_hop) | Refusal | Emergency | Total | Citation margin |
|---|---|---|---|---|---|
| Now, after the 2026-11-07 audit | 7 | 6 | 2 | 15 | ± 0.37 |
| 1 | 40 (20 + 20) | 10 | 10 | 60 | ± 0.15 |
| 2 | 70 (35 + 35) | 15 | 15 | 100 | ± 0.12 |
| **3, goal** | **100 (50 + 50)** | **15** | **15** | **130** | **± 0.10** |

The "Now" row assumes the audit's drop list without h22 and h24, which the owner is keeping for the time being.

The 100 answerable rows split by document, in proportion to the earlier 70-row plan:

| Document | Rows |
|---|---|
| Nghị định 87/2018, the core of LPG trade | 29 |
| Nghị định 105/2025, PCCC and insurance | 21 |
| Luật PCCC 2024 | 14 |
| Nghị định 17/2020, amendments | 11 |
| QCVN 04:2013, cylinders | 11 |
| QCVN 10:2012, stations | 8 |
| Thông tư 36/2025 | 6 |

A multi-hop row may span two documents; count it under the document of its first source. The 69 stub documents, which are only referenced, are good material for refusal rows.

**What it costs.** The first audit kept only 7 of 15 answerable rows. If drafts keep failing the audit at about that rate, 93 more answerable rows means writing about 200 drafts, or roughly 20 web-chat batches of 10, each followed by the audit in runbook §5. Stages 1 and 2 are each useful on their own, so stop at any stage if the effort stops being worth it. A wrong label costs more than a missing row.

## 6. Reading a holdout report with this in mind

- **Put a margin on every rate.** With 40 answered rows, a citation accuracy of 0.60 means "somewhere from 0.45 to 0.75", not "0.60".
- **Two runs of the same system differ by chance.** LLM sampling alone moved gold rows between runs in this project. Before calling a change real, the difference should clear section 3.3's bar for the holdout's size.
- **When gold and the holdout disagree, use section 3.2's test.** A gap that passes it means the gold number is over-fitted, and gold-only tuning should stop until the cause is found.
- **Category breakdowns have their own, smaller n.** Multi-hop citation 0.333 on the first run was 3 of 9 rows: a margin of ±0.31.

## 7. Computing it yourself

Python's standard library is enough. In a terminal:

```bash
python3 -c '
from math import sqrt
z = 1.96
def margin(p, n): return z * sqrt(p * (1 - p) / n)
def rows_for(E, p=0.5): return z * z * p * (1 - p) / E ** 2
def wilson(k, n):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)
print("margin, 7 of 14:", round(margin(0.5, 14), 3))
print("Wilson interval, 7 of 14:", wilson(7, 14))
print("rows for +-0.10:", round(rows_for(0.10)))
print("paired rows, gap 0.10, 20% change:", round((1.96 + 0.84) ** 2 * 0.20 / 0.10 ** 2))
'
```

It prints:

```
margin, 7 of 14: 0.262
Wilson interval, 7 of 14: (0.268, 0.732)
rows for +-0.10: 96
paired rows, gap 0.10, 20% change: 157
```

Use the **Wilson interval** in place of `p ± E` when n is small (under about 30) or p is near 0 or 1. The simple formula then gives impossible ranges. For example, 2 of 2 emergency rows caught gives `1.0 ± 0`, while the Wilson interval is 0.34 to 1.0, which is honest about how little 2 rows show.

## Sources

- Sample size for a proportion, confidence intervals and the Wilson interval: any introductory statistics text. NIST/SEMATECH e-Handbook of Statistical Methods, §7.2.4: https://www.itl.nist.gov/div898/handbook/prc/section2/prc24.htm
- McNemar's test for paired comparisons: https://en.wikipedia.org/wiki/McNemar%27s_test
- The rule of three: https://en.wikipedia.org/wiki/Rule_of_three_(statistics)
- The design effect: https://en.wikipedia.org/wiki/Design_effect
- RAG evaluation set sizes in practice:
  - https://theneuralbase.com/ragas/learn/beginner/dataset-size-recommendations/
  - https://www.beri.net/article/rag-evaluation-dataset-200-hand-written-questions-synthetic-ragas-deepeval-langfuse-2026
