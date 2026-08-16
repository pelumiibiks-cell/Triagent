

from __future__ import annotations

import csv
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import classifier  # noqa: E402
import safety  # noqa: E402
from data_layer import Dataset, load_dataset  # noqa: E402
from context_builder import build_all  # noqa: E402
from media_analysis import attach_to_contexts, load_cache  # noqa: E402

HEADER = ["message_id", "action", "message_type", "reason", "confidence",
          "evidence_message_ids"]

SAMPLE_OUTPUT = os.path.join(config.ROOT, "output_sample.csv")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def build_rows(ds: Dataset, sample: bool = False) -> List[Dict[str, Any]]:
    """Classifications + safety overrides, in messages.csv source order."""
    rows = ds.sample_messages if sample else ds.messages
    ctxs = build_all(ds, rows)
    attach_to_contexts(ctxs, load_cache())

    path = (classifier.DRYRUN_CHECKPOINT if sample
            else classifier.CHECKPOINT_PATH)
    results = safety._load_results(path)
    if not results:
        raise RuntimeError(
            "No classifications found at %s.\nRun: python code/classifier.py%s"
            % (path, " --dry-run" if sample else ""))

    final = safety.apply_all(ctxs, results)

    missing = [c.message_id for c in ctxs if c.message_id not in final]
    if missing:
        raise RuntimeError(
            "%d message(s) have no classification, so output.csv would be "
            "incomplete: %s\nRe-run the classifier to fill them in."
            % (len(missing), missing[:10]))

    out: List[Dict[str, Any]] = []
    for ctx in ctxs:                       # source order, one row per message
        r = final[ctx.message_id]
        ids = r.get("evidence_message_ids") or []
        out.append({
            "message_id": ctx.message_id,
            "action": r["action"],
            "message_type": r["message_type"],
            "reason": " ".join(str(r["reason"]).split()),
            "confidence": "%.2f" % float(r["confidence"]),
            "evidence_message_ids": ";".join(ids) if ids else "none",
        })
    return out


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write with the csv module and minimal quoting -- never string concat."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)



# validation -- re-reads the written file from disk


def validate_file(path: str, ds: Dataset, sample: bool = False
                  ) -> Tuple[bool, List[str]]:
    """Independent validation of the file on disk. Returns (ok, failures)."""
    failures: List[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    if not os.path.isfile(path):
        return False, ["output file does not exist: %s" % path]

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    # header, exact order
    if header != HEADER:
        fail("header mismatch.\n  expected: %s\n  got     : %s" % (HEADER, header))

    source = ds.sample_messages if sample else ds.messages
    expected_ids = [m["message_id"] for m in source]

    # -- one row per message, no dupes, no extras, no missing --
    got_ids = [r["message_id"] for r in rows]
    if len(rows) != len(expected_ids):
        fail("row count %d != %d rows in source" % (len(rows), len(expected_ids)))
    dupes = {i for i in got_ids if got_ids.count(i) > 1}
    if dupes:
        fail("duplicate message_ids: %s" % sorted(dupes)[:10])
    missing = set(expected_ids) - set(got_ids)
    extra = set(got_ids) - set(expected_ids)
    if missing:
        fail("%d message_ids missing from output: %s"
             % (len(missing), sorted(missing)[:10]))
    if extra:
        fail("%d message_ids in output that are not in source: %s"
             % (len(extra), sorted(extra)[:10]))
    if set(got_ids) != set(expected_ids):
        fail("id set does not match source exactly")

    history_ids = set(ds.history)
    batch_ids = {m["message_id"] for m in ds.messages}
    sample_ids = {m["message_id"] for m in ds.sample_messages}

    for r in rows:
        mid = r["message_id"]

        if r["action"] not in classifier.ACTIONS:
            fail("%s: action %r not in %s" % (mid, r["action"], classifier.ACTIONS))
        if r["message_type"] not in classifier.MESSAGE_TYPES:
            fail("%s: message_type %r not allowed" % (mid, r["message_type"]))

        if not (r["reason"] or "").strip():
            fail("%s: empty reason" % mid)

        try:
            c = float(r["confidence"])
        except (TypeError, ValueError):
            fail("%s: confidence %r does not parse as float" % (mid, r["confidence"]))
        else:
            if not 0.0 <= c <= 1.0:
                fail("%s: confidence %s outside [0,1]" % (mid, c))

        ev = (r["evidence_message_ids"] or "").strip()
        if not ev:
            fail("%s: evidence_message_ids is blank (must be ids or 'none')" % mid)
        elif ev != "none":
            for e in ev.split(";"):
                e = e.strip()
                if not e:
                    fail("%s: empty id in evidence list %r" % (mid, ev))
                elif e in batch_ids or e in sample_ids:
                    fail("%s: evidence id %r comes from the message batch, not "
                         "message_history.csv" % (mid, e))
                elif e not in history_ids:
                    fail("%s: evidence id %r does not exist in message_history.csv"
                         % (mid, e))

    return not failures, failures


# main


def _main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    sample = "--dry-run" in sys.argv
    validate_only = "--validate-only" in sys.argv
    path = SAMPLE_OUTPUT if sample else config.OUTPUT_CSV

    print("=" * 74)
    print("PHASE 7 - OUTPUT WRITER + VALIDATOR%s"
          % ("  [DRY RUN]" if sample else ""))
    print("=" * 74)

    ds = load_dataset()

    if not validate_only:
        rows = build_rows(ds, sample=sample)
        write_csv(rows, path)
        print("wrote %d rows -> %s" % (len(rows), path))

    print("\nVALIDATING %s (re-read from disk)" % path)
    ok, failures = validate_file(path, ds, sample=sample)
    if not ok:
        print("\nVALIDATION FAILED - %d problem(s):" % len(failures))
        for f in failures[:40]:
            print("  - %s" % f)
        return 1

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    source = ds.sample_messages if sample else ds.messages

    print("  [PASS] header exactly %s" % HEADER)
    print("  [PASS] %d rows, one per message, ids match source exactly"
          % len(rows))
    print("  [PASS] every action in %s" % (classifier.ACTIONS,))
    print("  [PASS] every message_type in the allowed set")
    print("  [PASS] every confidence parses as float in [0,1]")
    print("  [PASS] no empty reason")
    print("  [PASS] every evidence id exists in message_history.csv")
    print("  [PASS] no evidence id drawn from the message batch")

    # the assertion AGENTS.md names explicitly
    assert ({r["message_id"] for r in rows}
            == {m["message_id"] for m in source}), "id set mismatch"
    print("  [PASS] set(output.message_id) == set(source.message_id)")

    da: Dict[str, int] = {}
    dt: Dict[str, int] = {}
    for r in rows:
        da[r["action"]] = da.get(r["action"], 0) + 1
        dt[r["message_type"]] = dt.get(r["message_type"], 0) + 1
    print("\n  action distribution      : %s" % dict(sorted(da.items())))
    print("  message_type distribution: %s" % dict(sorted(dt.items())))
    confs = [float(r["confidence"]) for r in rows]
    print("  confidence: min=%.2f max=%.2f mean=%.3f"
          % (min(confs), max(confs), sum(confs) / len(confs)))
    wl = [len(r["reason"].split()) for r in rows]
    print("  reason words: min=%d max=%d mean=%.1f" % (min(wl), max(wl),
                                                       sum(wl) / len(wl)))
    nnone = sum(1 for r in rows if r["evidence_message_ids"] == "none")
    ncited = sum(len(r["evidence_message_ids"].split(";"))
                 for r in rows if r["evidence_message_ids"] != "none")
    print("  evidence: %d ids cited, %d rows cite 'none'" % (ncited, nnone))

    # round-trip proof: re-parsing yields identical field values
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    print("\n  file size: %d bytes, %d physical lines"
          % (len(raw), len(raw.splitlines())))
    embedded = sum(1 for r in rows if "\n" in r["reason"])
    print("  reasons containing newlines: %d (must be 0 for a clean CSV)"
          % embedded)

    if sample:
        gold = {r["message_id"]: r for r in ds.sample_messages}
        a = sum(1 for r in rows if r["action"] == gold[r["message_id"]]["action"])
        t = sum(1 for r in rows
                if r["message_type"] == gold[r["message_id"]]["message_type"])
        n = len(rows)
        print("\n  DRY-RUN ACCURACY vs gold: action %d/%d (%.0f%%)  "
              "type %d/%d (%.0f%%)" % (a, n, 100.0 * a / n, t, n, 100.0 * t / n))

    print("\n  first 3 rows as written:")
    for line in raw.splitlines()[:4]:
        print("    %s" % line[:150])

    print("\n" + "=" * 74)
    print("OUTPUT VALID - %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
