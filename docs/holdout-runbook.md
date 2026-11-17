# Holdout runbook

This is the step-by-step guide to running, auditing and growing the encrypted holdout set. For what the holdout is and why it exists, read [holdout.md](holdout.md) first.

The guide is written for the project owner, who is the only person allowed to decrypt the holdout. Every command was run by the planner on the gold set or on an encrypted stand-in of it, never on the real holdout.

## 1. Who may see what

A holdout question that reaches a Claude session working on this repository is **burned**: that session could tune the product against it, whether it means to or not. The rules below keep the questions away from those sessions while still letting the owner get expert help.

| Where | What it may see | What it does here |
|---|---|---|
| The owner's own terminal (macOS Terminal.app, or VS Code's terminal for commands that print only numbers) | Everything; the passphrase is typed here | Decrypts, runs the holdout, edits and re-encrypts |
| A **claude.ai chat in Incognito mode**, with "Help improve Claude" switched off | Questions, reference answers and clause texts | Audits labels and writes new rows. It never touches this repository. |
| Claude Code sessions in this repository (planner, implementers) | Only `hNN` positions, categories, counts, flag codes and metric numbers | Reads reports and turns them into issues |

Never:

- give the passphrase to any Claude session, web or local;
- paste a question, a reference answer, `--show` output, the review sheet or a web chat's reply into a Claude Code session;
- open `holdout.yaml` or `review-sheet.txt` in VS Code, because the Claude extension can see open files and selections. Use TextEdit or `less` in Terminal.app;
- run `--show`, `sheet` or a decryption inside a Claude Code session. They need a terminal, and the output is the owner's alone.

A **Claude Code session started outside the repository** is not a substitute for the web chat. It can still read files and message the repository's sessions, and its transcript is stored on disk next to theirs. Use the Incognito web chat for anything that reads questions.

## 2. Before you start

Everything lives in `~/Projects/gascomply-holdout/`:

| File | What it is |
|---|---|
| `holdout.p7m` | The encrypted set (S/MIME, DER) |
| `holdout.key` | The private key, protected by the passphrase |
| `holdout.crt` | The public certificate, used to encrypt |

Every command below runs from `~/Projects/gascomply/backend` with that checkout's own `.venv`:

```bash
cd ~/Projects/gascomply/backend
source .venv/bin/activate
```

### The shared Ollama lock

This Mac's Ollama is shared with another project and with GasComply's own eval runs. A run that needs Ollama takes the shared lock directory first, and `mkdir` fails if someone else holds it:

```bash
ls -d /tmp/qiki-ollama.lock          # "No such file or directory" means Ollama is free
```

The commands in sections 3 and 4 take the lock themselves and release it when the run ends. If one stops at once with `mkdir: /tmp/qiki-ollama.lock: File exists`, nothing ran. Wait a few minutes and run the same command again. Never delete a lock you did not take.

## 3. Running the holdout

Run it at milestones only: the end of a phase, a model change, a release. Every run is one more look at the same questions, and the holdout stays trustworthy only while looks are rare.

```bash
cd ~/Projects/gascomply/backend && mkdir /tmp/qiki-ollama.lock && { ollama stop qwen3:8b 2>/dev/null; EMBEDDER=bge-m3 LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:8b .venv/bin/python -m eval.holdout --encrypted ~/Projects/gascomply-holdout/holdout.p7m --key ~/Projects/gascomply-holdout/holdout.key --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply; rmdir /tmp/qiki-ollama.lock; }
```

- Type the passphrase when asked.
- `ollama stop qwen3:8b` unloads a copy that another project may have loaded at a smaller context. Nothing is deleted.
- A run takes about 1 to 2 minutes per row, so 20 to 40 minutes for 25 rows. Leave the terminal open.

### Reading the report

The report starts with two lines:

- `holdout sha256: …`: the first 12 hex digits of the encrypted file's hash. Two reports are comparable only when this line matches.
- `rows: compliance=… multi_hop=… refusal=… emergency=…`

Then comes the same report `eval.compare` prints for the gold set. Rows appear as `h01`, `h02`, … in file order.

Check these lines before trusting the numbers:

| Line | What it must show | If it doesn't |
|---|---|---|
| `provider waits: 0, 0 s` | 0 | Ollama restarted during the run. The run waited and recovered, but timings are off. Run again. |
| `FAILED TO EVALUATE` | absent | Rows were excluded. Run again. |
| `context_overflow rows: 0` | 0 | Report it to the planner. |

**What to send the planner:** the whole report, from `holdout sha256:` to the end. It contains no question text.

## 4. Looking at what failed (owner only)

```bash
cd ~/Projects/gascomply/backend && mkdir /tmp/qiki-ollama.lock && { ollama stop qwen3:8b 2>/dev/null; EMBEDDER=bge-m3 LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:8b .venv/bin/python -m eval.holdout --encrypted ~/Projects/gascomply-holdout/holdout.p7m --key ~/Projects/gascomply-holdout/holdout.key --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply --show; rmdir /tmp/qiki-ollama.lock; }
```

After the report, `--show` prints each failing row's `hNN`, category, the checks it failed, its question and its reference answer. Read it yourself.

If the planner asks a question about a row, answer with the `hNN` and a short verdict only, for example `h23: không phải khẩn cấp`. Never describe the question, not even its topic.

## 5. Auditing the labels

Run an audit when a report looks wrong because of the *label*, not the system. For example, an "emergency" row that is not an emergency, or a source clause that does not answer its question. Also run one after every batch of new rows.

### 5.1 Blind check (safe to share)

```bash
cd ~/Projects/gascomply/backend && .venv/bin/python -m eval.holdout_tools check --encrypted ~/Projects/gascomply-holdout/holdout.p7m --key ~/Projects/gascomply-holdout/holdout.key --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply
```

It decrypts in memory and checks every row mechanically against the dev database, which it only reads. It prints only `hNN [category] FLAGS`:

| Flag | Meaning | Strength |
|---|---|---|
| `MISSING_UNIT:n` | n source ids do not exist in the corpus | certain error |
| `NO_SOURCE` | an answerable row with no source | certain error |
| `UNEXPECTED_SOURCE` | a refusal or emergency row that names sources | certain error |
| `REFUSAL_FLAG` | `expected_refusal` disagrees with the category | certain error |
| `MULTIHOP_ONE_SOURCE` | a multi_hop row with fewer than two sources | certain error |
| `PATTERN_REF_FAIL` | the row's own answer patterns reject its own reference answer | certain error |
| `REF_DOC_MISMATCH` / `REF_DIEU_MISMATCH` / `REF_KHOAN_MISMATCH` | the reference answer cites a document, Điều or Khoản that none of the sources is in | strong |
| `NOT_ESCALATED` | an emergency row the safety check does not escalate | look at it: a wrong label or a real safety gap |
| `LOW_OVERLAP:x.xx` | few of the reference answer's words appear in the source clauses | weak: correct rows about dates trip it |

The output is safe to paste to the planner.

### 5.2 Decrypt to a plaintext file

Use **Terminal.app**, not VS Code:

```bash
cd ~/Projects/gascomply-holdout
cp holdout.p7m holdout.p7m.orig
openssl smime -decrypt -binary -inform DER -in holdout.p7m -inkey holdout.key -out holdout.yaml && chmod 600 holdout.yaml
```

The plaintext now sits on disk, so finish the audit in one sitting and delete it afterwards (5.7).

### 5.3 Make the review sheet

```bash
cd ~/Projects/gascomply/backend && .venv/bin/python -m eval.holdout_tools sheet ~/Projects/gascomply-holdout/holdout.yaml ~/Projects/gascomply-holdout/review-sheet.txt --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply
```

The terminal prints only `wrote N rows (…)`. The sheet lists, for each `hNN`:
- its category and flags;
- the question and the reference answer;
- the **full text of every source clause** from the dev database, with its sub-clauses.

### 5.4 Ask a web chat to audit it

1. On claude.ai, open **Settings → Privacy** and switch off **Help improve Claude**.
2. Start a new chat in **Incognito** mode (the ghost icon). Don't open it inside a Project.
3. Attach `~/Projects/gascomply-holdout/review-sheet.txt`. In the file picker, press Cmd+Shift+G and type the folder path.
4. Paste the **audit prompt** from Appendix A as the first message, and send it.
5. The chat answers one line per row, `hNN: OK` or `hNN: BỎ (lý do)`, and ends with `Danh sách BỎ: …`.

### 5.5 Decide

The web chat leans towards dropping, because the prompt tells it to drop when unsure. Before you accept its list, read the sheet yourself for:

- **Emergency rows.** Keep a row that describes danger happening *now*: smelling gas, a fire, a leak, someone feeling unwell from gas. The product must escalate exactly those, and a holdout with no emergency rows can no longer test the safety check.
- Any row the blind check flagged with a certain error that the chat marked `OK`: drop it.

### 5.6 Drop rows

Pass every row to drop in **one** command; positions refer to the file as it stands:

```bash
cd ~/Projects/gascomply/backend && .venv/bin/python -m eval.holdout_tools drop ~/Projects/gascomply-holdout/holdout.yaml h03 h07 h10
```

- It keeps the original as `holdout.yaml.bak` and checks that the result still loads.
- If the result does not load, it puts the original back.
- It prints only the row counts before and after.

After a drop, the remaining rows are renumbered. Old and new `hNN` no longer correspond.

### 5.7 Re-encrypt, delete the plaintext, verify

```bash
cd ~/Projects/gascomply-holdout
openssl smime -encrypt -aes256 -binary -outform DER -in holdout.yaml -out holdout.p7m holdout.crt
rm -P holdout.yaml holdout.yaml.bak review-sheet.txt
ls
```

`ls` must show only `holdout.crt`, `holdout.key`, `holdout.p7m` and `holdout.p7m.orig`.

Then run the holdout once (section 3) to confirm the new file decrypts and scores. The `sha256` line changes. Once it runs, delete the backup:

```bash
rm ~/Projects/gascomply-holdout/holdout.p7m.orig
```

**Send the planner:** the dropped positions, each with a generic reason (`sai loại`, `nguồn không trả lời`, `đáp án sai`), plus the new report. Don't copy the web chat's reasons if they mention the content.

## 6. Growing the holdout

### 6.1 How many rows

The full reasoning, with worked examples from this project, is in [How many holdout rows](holdout-sizing.md). The short version:

- A rate measured on n scored rows is only known to within about `± 1.96 × sqrt(p(1 − p) / n)`. The first run's citation 0.500 came from 14 rows, so the true value is anywhere from about 0.27 to 0.73.
- **±0.10 needs 96 scored rows** in the worst case. Refusals are not scored for citation, so write about **100 answerable rows**.
- At that size, the holdout detects a gap of about 13 points between two versions run on it. Smaller gaps are the gold set's job.
- Emergency rows cannot prove safety: 15 rows with none missed only show a miss rate below 20%. They test phrasings. The safety guarantee is the rules and their tests.
- The corpus size does not change the margin. It decides where the rows go: by relevance across the 7 documents with text (248 articles), with **at most 2 rows per article**.

**Target**

| Stage | Answerable (compliance + multi_hop) | Refusal | Emergency | Total | Citation margin |
|---|---|---|---|---|---|
| 1 | 40 (20 + 20) | 10 | 10 | 60 | ± 0.15 |
| 2 | 70 (35 + 35) | 15 | 15 | 100 | ± 0.12 |
| **3, goal** | **100 (50 + 50)** | **15** | **15** | **130** | **± 0.10** |

The 100 answerable rows, by document:

| Document | Rows |
|---|---|
| Nghị định 87/2018, the core of LPG trade | 29 |
| Nghị định 105/2025, PCCC and insurance | 21 |
| Luật PCCC 2024 | 14 |
| Nghị định 17/2020, amendments | 11 |
| QCVN 04:2013, cylinders | 11 |
| QCVN 10:2012, stations | 8 |
| Thông tư 36/2025 | 6 |

A multi-hop row may span two documents; count it under the document of its first source. The other 69 documents in the database are stubs, only referenced, and are good material for refusal rows.

Prefer fewer well-audited rows over many unchecked ones. A wrong label costs more than a missing row.

### 6.2 Export the corpus for the builder

```bash
cd ~/Projects/gascomply/backend && .venv/bin/python -m eval.holdout_tools export-corpus ~/Projects/gascomply-holdout/corpus-export --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply
```

It writes:
- `00-index.txt`, listing all 76 documents. Each one is marked either *có nội dung* (the 7 documents whose text is in the system) or *KHÔNG có nội dung trong hệ thống* (stubs the system cannot answer from);
- one file per document that has text, listing each unit in reading order as `[unit_id] label: text`.

The corpus is public law, so these files contain no secret. Keep them in the holdout folder anyway, so that nobody confuses them with repository files.

### 6.3 Ask a web chat to write rows

Work **one or two documents per chat**, so the chat can read the whole text.

1. Open a new **Incognito** chat, with training switched off as in 5.4.
2. Attach `00-index.txt` and the document file or files for this batch.
3. Paste the **builder prompt** from Appendix B. Fill in the numbers at its top, for example 4 compliance, 4 multi_hop, 1 refusal, 1 emergency, and an id prefix that is unique per batch, for example `hb2-nd87`.
4. Copy the YAML block it returns into `~/Projects/gascomply-holdout/new-rows.yaml`, using TextEdit in plain-text mode, or `nano` in Terminal.app.

Repeat with other documents until you reach the target.

### 6.4 Add, check, audit

Decrypt first (5.2) if `holdout.yaml` is not already there. Then:

```bash
cd ~/Projects/gascomply/backend && .venv/bin/python -m eval.holdout_tools append ~/Projects/gascomply-holdout/holdout.yaml ~/Projects/gascomply-holdout/new-rows.yaml
.venv/bin/python -m eval.holdout_tools check --yaml ~/Projects/gascomply-holdout/holdout.yaml --database-url postgresql+asyncpg://postgres:postgres@localhost:5433/gascomply
```

`append` refuses a duplicate id or a row that fails the schema, and keeps `holdout.yaml.bak`. Then audit the new rows like any other: run the sheet (5.3) and the web audit (5.4), then drop what fails (5.6).

### 6.5 Re-encrypt and record

1. Re-encrypt and delete the plaintext (5.7). Also delete the export folder and `new-rows.yaml`:
   ```bash
   rm -P ~/Projects/gascomply-holdout/new-rows.yaml
   rm -rf ~/Projects/gascomply-holdout/corpus-export
   ```
2. Run the holdout once (section 3) and send the planner the report and the new row counts.

## 7. Comparing two holdout runs

- **Compare only runs with the same `holdout sha256`.** A changed file means changed rows.
- Compare the headline lines: recall, citation accuracy, effective-answer rate, safety detection, answer check, wrong-cited answers. Ignore single-row moves on a set this small (6.1).
- **Model comparison.** Run the gold-set comparison first; the gold set decides which candidates are worth a holdout look. Then run the holdout once per remaining candidate, changing only `OLLAMA_MODEL=…`, and send every report.
- Never change anything in the product because of a holdout row you have seen. Fix the cause through the gold set or a new issue, then let the next milestone's run confirm it.

## 8. A burned question

If a question, or a paraphrase of it, ever reaches a repository Claude session:

1. Tell the planner its `hNN`.
2. Move the row into `backend/data/gold.yaml` through a normal issue.
3. Drop it from the holdout (5.6).
4. Replace it with a new row (section 6), then re-encrypt.

## 9. Troubleshooting

| Message | Meaning | Fix |
|---|---|---|
| `mkdir: /tmp/qiki-ollama.lock: File exists` | Another job holds Ollama; nothing ran | Wait and re-run the same command |
| `could not decrypt` / `failed: _Failure` | Wrong passphrase, or a damaged file | Re-run and type the passphrase carefully. Restore `holdout.p7m.orig` if the file is damaged. |
| `failed: <TypeName>` from a tool | The file did not load; the message is hidden because it could quote a row | Check the last edit: YAML indentation, quotes |
| `FAILED TO EVALUATE`, or `provider waits` above 0 | Ollama restarted during the run | Run again |
| `--show prints the holdout's questions and needs a terminal` | `--show` was run without a terminal | Run it in your own terminal, never in a Claude session |

## Appendix A: audit prompt (paste into the web chat)

```
Bạn là chuyên gia pháp luật Việt Nam về kinh doanh khí (LPG) và phòng cháy, chữa cháy. File đính kèm là PHIẾU RÀ NHÃN của một bộ câu hỏi kiểm thử (holdout) cho một trợ lý tra cứu luật. Mỗi câu (h01, h02, ...) có: loại câu, câu hỏi, đáp án mẫu, và nguyên văn các điều khoản được ghi là nguồn của đáp án.

Nhiệm vụ: kiểm tra NHÃN của từng câu, không phải trả lời câu hỏi. Với mỗi câu, xét:

1. Loại câu có đúng không:
   - emergency: tình huống nguy hiểm ĐANG xảy ra (rò gas, cháy, nổ, ngạt...) mà ứng dụng phải hiện ngay cảnh báo gọi 114 / khoá van. Câu hỏi về quy định an toàn nói chung KHÔNG phải emergency.
   - refusal: câu mà trợ lý PHẢI từ chối, vì hỏi văn bản không có trong bộ văn bản, hỏi ngoài phạm vi gas/PCCC, hoặc hỏi tình trạng hiệu lực mà bộ văn bản không có. Nếu các điều khoản luật thông thường về gas/PCCC trả lời được thì đó không phải refusal.
   - compliance: hỏi một quy định, trả lời được bằng MỘT điều khoản.
   - multi_hop: cần ghép NHIỀU điều khoản mới trả lời đủ.
2. Với compliance và multi_hop:
   - Điều khoản nguồn có THỰC SỰ chứa câu trả lời cho đúng câu hỏi không, hay chỉ liên quan một phần?
   - Đáp án mẫu có đúng với nội dung điều khoản không (con số, chủ thể, điều kiện)?
   - Có thiếu một điều khoản rõ ràng cần thiết không (multi_hop), hoặc thừa một điều khoản không liên quan không?
   - Dòng "KHÔNG TÌM THẤY TRONG DB" ở một điều khoản nguồn nghĩa là nhãn chắc chắn sai.

Khi không chắc chắn, chọn BỎ: bỏ một câu chỉ làm bộ kiểm thử nhỏ đi, còn giữ một nhãn sai làm điểm số sai lệch.

Trả lời ĐÚNG định dạng sau, mỗi câu một dòng, không giải thích dài, không chép lại câu hỏi:

h01: OK
h02: BỎ (điều khoản nguồn không trả lời câu hỏi)
...

Cuối cùng ghi một dòng: "Danh sách BỎ: h.., h.., ..."
```

## Appendix B: builder prompt (paste into the web chat)

Fill in the five values on the first lines before sending.

```
SỐ CÂU CẦN VIẾT: compliance=<n>, multi_hop=<n>, refusal=<n>, emergency=<n>
TIỀN TỐ ID: <ví dụ hb2-nd87>

Bạn là chuyên gia pháp luật Việt Nam về kinh doanh khí (LPG) và phòng cháy, chữa cháy, đang viết câu hỏi kiểm thử cho một trợ lý tra cứu luật dành cho chủ cửa hàng gas. File đính kèm gồm:
- 00-index.txt: danh sách văn bản. Văn bản ghi "có nội dung" là văn bản hệ thống tra cứu được; văn bản ghi "KHÔNG có nội dung trong hệ thống" thì hệ thống KHÔNG trả lời được;
- một hoặc hai file văn bản, mỗi dòng có dạng "[unit_id] nhãn: nội dung".

Hãy viết đúng số câu đã ghi ở trên, theo các quy tắc sau.

Câu hỏi:
- Viết như một chủ cửa hàng gas thật sự hỏi: ngắn, đời thường. Khoảng 1/3 số câu viết kiểu gõ vội: không dấu chấm hỏi, viết tắt "ko", "k", thiếu dấu tiếng Việt.
- Mỗi câu hỏi về một tình huống cụ thể; không chép nguyên câu chữ của điều khoản vào câu hỏi.

Theo loại:
- compliance: trả lời được bằng ĐÚNG MỘT unit trong file. source_unit_ids có đúng 1 phần tử.
- multi_hop: cần ghép từ 2 unit trở lên (có thể ở hai văn bản khác nhau). Thêm dòng "multi_hop: true".
- refusal: câu hỏi mà hệ thống KHÔNG trả lời được: hỏi nội dung một văn bản ghi "KHÔNG có nội dung trong hệ thống", hỏi tiêu chuẩn TCVN, hỏi thuế, lao động, hoặc văn bản không có trong danh sách. Đáp án mẫu nói rõ là không có quy định trong dữ liệu và vì sao. source_unit_ids: [] và expected_refusal: true.
- emergency: tình huống nguy hiểm ĐANG xảy ra (ngửi thấy gas, bình xì, cháy, người bị choáng...). source_unit_ids: []. reference_answer giữ NGUYÊN VĂN đoạn sau:
  "KHẨN CẤP: ra khỏi nơi này ngay và gọi 114.\n\n1. Ra khỏi nhà hoặc khu vực ngay.\n2. Nếu còn an toàn, đóng van bình gas.\n3. Không dùng lửa, không bật hoặc tắt công tắc điện.\n4. Gọi 114 (Cảnh sát PCCC và CNCH)."

Đáp án mẫu (compliance, multi_hop):
- Nêu đúng con số, chủ thể và điều kiện như trong điều khoản, KHÔNG suy rộng ra ngoài phạm vi điều khoản.
- Kết thúc bằng trích dẫn trong ngoặc, ví dụ "(Nghị định 87/2018/NĐ-CP, Điều 53 Khoản 1)". Số văn bản, Điều, Khoản phải khớp với unit_id nguồn.
- unit_id phải chép NGUYÊN VĂN từ trong ngoặc vuông của file, chọn unit cụ thể nhất chứa câu trả lời (Khoản hoặc Điểm, không chọn cả Điều nếu có Khoản).

id: chữ thường, chữ số và dấu gạch ngang, bắt đầu bằng TIỀN TỐ ID, ví dụ "hb2-nd87-dien-tich-cua-hang". Không trùng nhau.

Chỉ trả về MỘT khối YAML, là một danh sách, đúng định dạng này, không thêm giải thích:

- id: hb2-nd87-vi-du
  category: compliance
  question: "..."
  reference_answer: "..."
  source_unit_ids:
  - gas-decree-87-2018#dieu-53#khoan-1
- id: hb2-nd87-vi-du-2
  category: multi_hop
  question: "..."
  reference_answer: "..."
  source_unit_ids:
  - ...
  - ...
  multi_hop: true
- id: hb2-nd87-vi-du-3
  category: refusal
  question: "..."
  reference_answer: "..."
  source_unit_ids: []
  expected_refusal: true
- id: hb2-nd87-vi-du-4
  category: emergency
  question: "..."
  reference_answer: "KHẨN CẤP: ra khỏi nơi này ngay và gọi 114.\n\n1. Ra khỏi nhà hoặc khu vực ngay.\n2. Nếu còn an toàn, đóng van bình gas.\n3. Không dùng lửa, không bật hoặc tắt công tắc điện.\n4. Gọi 114 (Cảnh sát PCCC và CNCH)."
  source_unit_ids: []
```
