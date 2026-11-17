# GasComply — Corpus Manifest (Phase 0)

Core Vietnamese legal-normative documents for the LPG-business + fire-safety (PCCC)
domain. Researched 2026-08-24, re-verified 2026-08-29, and **re-verified again
2026-09-01** — that last pass changed the list materially, see *Verification log*.

`corpus/manifest.json` is the machine-readable form of the table below and is the
only tracked file in `corpus/`; the documents themselves are downloaded locally and
never committed. **Re-verify status on vbpl.vn at download time** — the law changes,
and this table is only as good as its last check.

A row marked **not acquired** below is declared with `"acquired": false` in the
manifest (issue #59): it is part of the corpus and gets a `documents` row and a
graph node — so an edge naming it, e.g. `REPLACED_BY`, still resolves — but has no
file on disk, so `load_manifest` does not require one, and the ingest pipeline
skips it entirely (no units, no chunks, no embeddings). The field defaults to
`true`; a document without it must have its file present or ingest fails loudly,
which is still the right default for every document whose text is expected to
exist. Three rows are `acquired: false` today, each for a different reason
recorded in its own `meta.not_acquired_reason`: `lpg-store-tcvn-6223-2017` is
copyrighted and must be purchased (see *Licensing* below — this one is
permanent until someone buys it); `lpg-qcvn-08-2019` simply has not been
downloaded yet; `lpg-qcvn-08-2012` is expired and kept only so its `REPLACED_BY`
edge to QCVN 08:2019 resolves — its text is never needed.

## Documents

| id | title (VN, official) | number | type | issued | effective | status | acquired | notes |
|----|----------------------|--------|------|--------|-----------|--------|----------|-------|
| `pccc-law-2024` | Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ | 55/2024/QH15 | law | 2024-11-29 | 2025-07-01 | in_force | yes | 8 chương, 55 điều; replaces the 2001/2013 PCCC law |
| `pccc-decree-105-2025` | Nghị định quy định chi tiết và biện pháp thi hành Luật PCCC & CNCH | 105/2025/NĐ-CP | decree | 2025-05-15 | 2025-07-01 | in_force | yes | GUIDED_BY ← 55/2024; replaces NĐ 136/2020, 83/2017, 50/2024, 78/2011. Two-part Gazette source (manifest `parts`): Số 717+718 (Điều 1-47, Phụ lục I-VI) + Số 719+720 (Phụ lục VII-VIII) — **fixed 2026-10-07 (#126)** |
| `pccc-circular-36-2025` | Thông tư quy định chi tiết Luật PCCC & CNCH và NĐ 105/2025 | 36/2025/TT-BCA | circular | 2025-05-15 | 2025-07-01 | in_force | yes | **added 2026-09-01** |
| `gas-decree-87-2018` | Nghị định về kinh doanh khí | 87/2018/NĐ-CP | decree | 2018-06-15 | 2018-08-01 | in_force | yes | LPG retail-store conditions; AMENDED_BY ← 17/2020 |
| `gas-decree-17-2020` | Nghị định sửa đổi các NĐ về điều kiện đầu tư kinh doanh thuộc quản lý của Bộ Công Thương | 17/2020/NĐ-CP | decree | 2020-02-05 | 2020-03-22 | in_force | yes | **added 2026-09-01** — amends 87/2018 |
| `lpg-store-tcvn-6223-2017` | Cửa hàng khí dầu mỏ hóa lỏng (LPG) — Yêu cầu chung về an toàn | TCVN 6223:2017 | tcvn | 2017 | 2017 | in_force | **no** | **copyrighted, must be purchased** — see *Licensing* |
| `lpg-station-qcvn-10-2012` | An toàn trạm cấp khí dầu mỏ hóa lỏng | QCVN 10:2012/BCT | qcvn | 2012-12-28 | 2012 | in_force | yes | issued via TT 49/2012/TT-BCT |
| `lpg-cylinder-qcvn-04-2013` | An toàn chai chứa khí dầu mỏ hóa lỏng bằng thép | QCVN 04:2013/BCT | qcvn | 2013-07-31 | 2013 | in_force | yes | issued via TT 18/2013/TT-BCT; **the real source of the cylinder-inspection rule** |
| `lpg-qcvn-08-2019` | Khí dầu mỏ hóa lỏng (LPG) | QCVN 08:2019/BKHCN | qcvn | 2019 | 2020-01-01 | in_force | **no** | **added 2026-09-01**; issued via TT 14/2019/TT-BKHCN; not yet downloaded |
| `lpg-qcvn-08-2012` | Khí dầu mỏ hóa lỏng (LPG) | QCVN 08:2012/BKHCN | qcvn | 2012-04-12 | 2012 | **expired** | **no** | REPLACED_BY → QCVN 08:2019; kept as a metadata-only expired node for history questions, text not needed |

## Verification log — 2026-09-01

Four findings, three of which change what gets ingested.

**NĐ 87/2018 is still in force; the replacement was never issued.** The Ministry of
Industry and Trade drafted a replacement decree and collected comments (deadline
2024-10-01), but it has not been promulgated. The "biggest corpus risk" flagged in the
earlier passes resolves in favour of keeping the document. Status is plain `in_force`,
**not** the earlier "in_force / amending" — under the A5.2 rule a document is superseded
iff it is the src of a `REPLACED_BY` edge **or** `status != in_force`, so writing
`amending` here would have made the effective-date filter silently drop the single most
important document in the corpus.

**NĐ 17/2020/NĐ-CP was missing entirely.** It amends 87/2018 and took effect 2020-03-22.
Ingesting 87/2018 without it means the corpus holds article text that has since been
amended, and the assistant would quote it confidently. Now listed.

**QCVN 08:2012/BKHCN was dead and listed as in force.** It was replaced by
QCVN 08:2019/BKHCN (TT 14/2019/TT-BKHCN) effective 2020-01-01. The 2019 version is now
the in-force entry; the 2012 one is kept with `status: expired` so the `REPLACED_BY`
edge exists and history questions still work.

**TT 48/2011/TT-BCT was the wrong document and has been dropped.** The earlier table
described it as "Kiểm định chai chứa LPG", but 48/2011/TT-BCT governs quality management
of group-2 goods generally. The cylinder-inspection requirement actually lives in
**QCVN 04:2013/BCT** (issued via TT 18/2013/TT-BCT), which covers design, manufacture,
import, repair, inspection, storage, transport and use of refillable welded steel LPG
cylinders of 0.5–150 L, and which points at TCVN 7832:2007 for the periodic inspection
cycle.

**Confidence:** the vbpl.vn *lược đồ* page for 87/2018 returned HTTP 404 during this
pass, so these conclusions come from secondary legal databases (luatvietnam,
thuvienphapluat, caselaw, lawnet) cross-checked against each other, plus the government
portal for 105/2025. They agree, but **the authoritative check is the lược đồ page of
each document on vbpl.vn — do it at download time.**

## Candidate, not yet in the manifest

| document | why it is a candidate | why it is not listed |
|---|---|---|
| TT 41/2011/TT-BCT — quản lý an toàn trong lĩnh vực khí dầu mỏ hóa lỏng | directly on point for LPG safety management; effective 2012-02-01 | its current status could not be confirmed — NĐ 87/2018 may have superseded parts of it. The manifest carries only documents whose status is known; add it once vbpl.vn confirms. |

## Licensing and terms of service

- **TCVN standards are copyrighted** and sold by Ủy ban Tiêu chuẩn Đo lường Chất lượng
  (vsqi.gov.vn). TCVN 6223:2017 must be **purchased**; third-party PDFs circulating on
  the web are unauthorised copies and must not be used. `corpus/` is git-ignored, so a
  lawfully purchased copy stays on the owner's machine and is never published by this
  repository.
- **QCVN regulations and all Luật / Nghị định / Thông tư are public legal-normative
  documents** and carry no such restriction.
- Source priority: **vbpl.vn (official gazette) → the issuing ministry's portal →
  commercial databases**. The commercial sites are research entry points, not corpus
  sources.
- **No scraping.** Documents are downloaded by hand; this repository reads only the
  local `corpus/` directory and contains no code that fetches from these sites.

## Format

`corpus/<id>.html` or `corpus/<id>.pdf`, chosen per document — whichever is cleanest.
**Prefer HTML.** The parser anchors on line starts (`^Điều`, `^\d+\.`, `^[a-zđ]\)`), and
PDF extraction breaks exactly that: lines wrap to column width rather than to sentences,
page headers and footers repeat, and some vbpl.vn documents are scans with no text layer
at all. vbpl.vn serves a full-text HTML view (`vbpq-toanvan.aspx?ItemID=…`) that keeps
paragraph boundaries intact.

`manifest.json` records the actual filename, so update `path` to match the extension of
what was really downloaded.

A document whose source spans more than one downloaded file (issue #126 — some Công báo
publications split a long decree across two issues) declares `parts` instead of `path`:
an ordered list of `{"path": ..., "source_url": ...}` entries, one per file, concatenated
in that order at ingest time. Exactly one of `path`/`parts` is required. See
`pccc-decree-105-2025`'s own entry above for a real example.

### `.doc` (Word 97-2003) is not an extraction format — convert it first

Công báo publishes some documents (e.g. `pccc-law-2024`, Luật 55/2024/QH15) only as
legacy Word 97-2003 (`.doc`). `ingest/extract.py` supports `.html` and `.pdf` only and
rejects `.doc`/`.docx` outright (build plan A6 issue #33) — deliberately: the only
Python libraries that read legacy `.doc` either wrap an external converter binary
themselves or are unmaintained, so "supporting `.doc`" in code would still mean shelling
out to that same binary at ingest time, just less visibly than doing it once by hand.
That trades a one-line manual step for a runtime dependency on an external program being
installed and on `PATH` wherever the pipeline runs — worse for a ~10-document corpus that
changes rarely, not better.

**Convert with LibreOffice before adding the document to the manifest**, then point
`path` at the resulting `.html`:

```sh
soffice --headless --convert-to html --outdir corpus corpus/<id>.doc
```

LibreOffice is installed on the owner's machine at
`/Applications/LibreOffice.app/Contents/MacOS/soffice` (add it to `PATH`, or invoke it
by full path).

This is not a fallback for lack of a better option — it is the *better* extraction path.
Measured on the seven real documents in `corpus/` as of this writing, `pccc-law-2024`'s
`.doc → HTML` conversion parsed to exactly 8 chapters and 55 articles (the law's
documented structure) with 1 skipped anchor, while every PDF-sourced document in the
same batch skipped 24 to 211 anchors. If a source ever offers both a `.doc` and a PDF,
prefer converting the `.doc`.

## Known cross-reference edges to seed

Curated in `backend/data/document_edges.json` (build plan A6, issue #63) — human-authored, not
extracted, the same review model as `requirements.json`. Seeded as `source=curated`,
`review_status=pending`; only a human approval through `ingest.review` makes any of these
traversable. Two of the five (marked below) can also be produced by the regex pass once both
documents' text is on disk; when both a curated and a regex-extracted edge exist for the same
`(src_id, dst_id, edge_type)`, the curated one's `source` wins on upsert (`ingest/graph.py`'s own
module docstring has the reasoning).

- `pccc-decree-105-2025` **GUIDED_BY** → `pccc-law-2024` (also producible by regex)
- `pccc-circular-36-2025` **GUIDED_BY** → `pccc-law-2024` (also producible by regex)
- `pccc-circular-36-2025` **GUIDED_BY** → `pccc-decree-105-2025`
- `gas-decree-87-2018` **AMENDED_BY** → `gas-decree-17-2020`
- `lpg-qcvn-08-2012` **REPLACED_BY** → `lpg-qcvn-08-2019` — the one edge with no other possible
  source: both endpoints are `acquired: false`, so no text on disk states the repeal

`ISSUED_VIA` is not an `EdgeType` and its targets (below) are not manifest documents — these are
provenance notes, not graph edges:

- `lpg-cylinder-qcvn-04-2013` issued via TT 18/2013/TT-BCT
- `lpg-station-qcvn-10-2012` issued via TT 49/2012/TT-BCT

The eight `req:* → concept:lpg-retail-store` edges (`APPLIES_TO`, requirement → concept, A5.2) are
seeded by the requirement cluster (`requirements.json`, issue #51), not here — a document-level
`gas-decree-87-2018 APPLIES_TO concept:lpg-retail-store` line previously stood here and contradicted
that direction.

## Requirement cluster to seed (target = `concept:lpg-retail-store`)

Each becomes a `Requirement` node linked back to its source article/clause once the
parser runs. **Every value below still needs its article pinned against the real text**
— they come from secondary summaries, not from the statute.

| requirement | value | source (verify article) |
|---|---|---|
| minimum store area | ≥ 12 m² | NĐ 87/2018 (check 17/2020 for amendment) |
| distance from ignition source | ≥ 3 m (unless fire-rated wall) | NĐ 87/2018 |
| fire-resistance rating | ≥ level II | NĐ 87/2018 |
| electrical equipment | explosion-proof, ≥ 1.5 m from cylinders | NĐ 87/2018 |
| display racks | empty cylinders only | NĐ 87/2018 |
| supply contract | ≥ 1 year with a qualified trader | NĐ 87/2018 |
| licensing time | 15 working days | NĐ 87/2018 |
| cylinder inspection | per QCVN 04:2013/BCT (→ TCVN 7832:2007 for the cycle) | QCVN 04:2013/BCT |

## Research entry points (fetched 2026-08-24, re-checked 2026-09-01)

- NĐ 105/2025 Gazette landing page: https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-105-2025-nd-cp-44912.htm
  — part 1 (Số 717+718, Điều 1-47 + Phụ lục I-VI): https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2025/5/44912/56374-1-2025717-718105-2025-nd-cp.pdf
  — part 2 (Số 719+720, Phụ lục VII-VIII): https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2025/5/44912/56377-1-2025719-720105-2025-nd-cp.pdf
  — **fixed 2026-10-07 (#126):** the manifest previously pointed at
  `datafiles.chinhphu.vn/.../105-ndcp.signed.pdf`, described as the "officially signed PDF";
  that file is a 181-page scan with no text layer (0 extractable characters per page) and
  cannot be ingested without OCR. The two Gazette-edition PDFs above are the real,
  text-bearing source, now declared as the document's `parts` in the manifest.
- TT 36/2025/TT-BCA full text (HTML): https://vbpl.vn/bocongan/Pages/vbpq-toanvan.aspx?ItemID=177834&dvid=316
- QCVN 04:2013/BCT + TT 18/2013: https://www.vietnamtradeportal.gov.vn/kcfinder/upload/files/QCVN%2004_2013_BCT%2018_2013_TT-BCT%20VN.pdf
- TCVN 6223:2017 official (purchase): https://tieuchuan.vsqi.gov.vn/tieuchuan/view?sohieu=TCVN+6223:2017
- Draft replacement decree for gas trading: https://vibonline.com.vn/du_thao/du-thao-nghi-dinh-ve-kinh-doanh-khi

> ⚠️ Prefer the **official gazette** (vbpl.vn) or the issuing ministry portal for the
> authoritative full text. Respect each site's ToS — public legal-normative documents only.
