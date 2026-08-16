# TRIAGENT

Routes every incoming WhatsApp message in `dataset/messages.csv` to one of
`notify`, `digest`, or `mute`, personalised to the receiving user, and writes
`output.csv`.

The system reasons over text, **images** (Gemini vision) and **voice notes**
(local Whisper transcription), retrieves the user's own history as evidence,
and applies deterministic safety overrides on top of the model's decision.

---

## Requirements

- Python 3.13 (developed and validated on 3.13.0)
- A free Gemini API key from <https://aistudio.google.com/apikey> — no credit
  card required
- ~150 MB disk for the Whisper model weights (downloaded automatically on
  first run)

## Setup

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

### Required environment variable

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | **Required.** Gemini API key for image analysis and classification. |

Supply it by creating a `.env` file in the **repo root** (the directory
containing `dataset/`):

```
GEMINI_API_KEY=your_key_here
```

`.env` is gitignored and is **not** included in this archive — supply your own.
The key is read via `python-dotenv` + `os.environ` only; it is never hardcoded
and never printed. If it is missing, every entry point fails immediately at
startup with an explicit message rather than part-way through a run.

## Expected layout

Run from the repo root. The code expects `dataset/` beside `code/`:

```
.
├── code/            # this archive
├── dataset/         # provided by the organiser (not included here)
│   ├── messages.csv
│   ├── media/{images,audio}/
│   └── ... 11 more CSVs
├── cache/           # created on first run
└── output.csv       # produced by step 4
```

## Running

Four steps, in order. Each is independently resumable and caches its work.

```bash
# 1. Verify the environment, key and model ids against the live API
python code/config.py

# 2. Analyse media: 20 images via Gemini vision + 13 voice notes via local
#    Whisper. Cached in cache/media_analysis.json; a re-run costs 0 API calls.
python code/media_analysis.py

# 3. Classify all 110 messages. Checkpointed per message to
#    cache/classifications.jsonl, so a crash or quota wall means "resume".
python code/classifier.py --dry-run     # optional: 30 solved samples, scored
python code/classifier.py               # the real batch

# 4. Write and validate output.csv
python code/write_output.py
```

Total cost of a cold run: **~123 Gemini calls** (20 vision + 103 classification;
7 messages are resolved by a deterministic pre-filter and need no call). That
sits comfortably inside the free tier's daily allowance. Wall clock is roughly
12 minutes, dominated by deliberate rate-limit pacing.

### Useful flags

| Command | Effect |
|---|---|
| `python code/classifier.py --dry-run` | Run the 30 solved sample rows and score against their known answers |
| `python code/classifier.py --limit N` | Classify only the next N unprocessed messages |
| `python code/classifier.py --stats` | Show checkpoint progress; makes no API calls |
| `python code/media_analysis.py --stats` | Show media cache state; makes no API calls |
| `python code/write_output.py --validate-only` | Re-validate an existing output.csv |
| `--force` | Ignore caches/checkpoints and redo the work |

Every module also runs standalone as its own self-check:

```bash
python code/data_layer.py        # dataset integrity: 13 tables, 18 joins
python code/context_builder.py   # context bundles for all 140 messages
python code/evidence.py          # evidence contract + retrieval quality
python code/safety.py            # safety-override audit and regression guards
```

---

## Architecture

| Module | Role |
|---|---|
| `config.py` | Paths, model ids, rate limits, API-key loading. Single source of truth. |
| `data_layer.py` | Loads all 13 CSVs into indexed lookups. Blank → `None` at the boundary; every foreign-key accessor is None-safe. |
| `context_builder.py` | Per-message context bundle: sender, recipient, group, business relationship, and the derived signals `domain_mismatch`, `opted_out`, `group_muted`, `in_dnd`, `direct_mention`. |
| `media_analysis.py` | Gemini vision per distinct image (structured schema); `faster-whisper` locally for voice. Cached by `media_id`. |
| `evidence.py` | Retrieves historical messages as evidence, ranked by sender match + IDF-weighted lexical overlap + recency, joined with the user's actual past reaction. Enforces the evidence contract. |
| `classifier.py` | Narrow deterministic pre-filter, then one structured Gemini call per message. Validates against the enums; checkpoints per message. |
| `safety.py` | Post-classification overrides: force `mute`/`scam` on clear fraud; allow `notify` for a genuine direct mention in a muted group. |
| `write_output.py` | Writes `output.csv` via the `csv` module, then re-reads it from disk and validates independently. |

### Design decisions worth knowing

- **Standard library only for data handling.** The dataset's sole missing-value
  shape is the empty string; pandas would coerce those to `NaN` and blur a
  distinction that matters. `csv` + dataclasses are ample at this scale.
- **Voice notes are transcribed locally**, not sent to Gemini, keeping the
  limited free-tier budget for the reasoning calls that need it.
- **Images are analysed once per distinct `image_id`**, not per message, and
  cached — several messages share an image.
- **The pre-filter is deliberately narrow.** It fires only on overwhelming
  impersonation (mismatched domain **and** unverified **and** account younger
  than 90 days **and** ≥20 recent reports). Broadening it would mute
  legitimate businesses the user simply has not transacted with.
- **Safety overrides ignore warning contexts.** Anti-fraud *warnings* use the
  same vocabulary as fraud. A society admin writing "don't use any payment
  link shared by residents" is protecting the user; matching is therefore done
  per sentence and suppressed in negated/warning contexts. The same trap
  appears in images — a bank's own anti-scam poster is full of the word
  "scammers" — and is handled by the structured vision extraction.
- **Evidence uses an IDF-weighted overlap coefficient**, not Jaccard. Jaccard
  divides by the union and so is length-sensitive: attaching OCR'd media text
  to one side inflated the union and crushed genuine matches.
- **Repeated history rows are cited, not deduplicated.** Repetition is
  frequently the justification for muting a sender, so collapsing duplicates
  destroys the evidence.

---

## Validation

`write_output.py` re-reads the written file from disk and fails loudly unless:

- the header is exactly `message_id,action,message_type,reason,confidence,evidence_message_ids`
- there is exactly one row per `message_id` in `dataset/messages.csv`, with no
  duplicates, none missing and none extra
- every `action` ∈ {`notify`,`digest`,`mute`} and every `message_type` is in
  the allowed set of 11
- every `confidence` parses as a float in [0,1]
- no `reason` is empty
- every `evidence_message_ids` entry exists in `message_history.csv`, and none
  is drawn from the message batch itself

Measured on the 30 solved rows in `sample_messages.csv`:
**action accuracy 30/30 (100%), message_type accuracy 27/30 (90%).**
