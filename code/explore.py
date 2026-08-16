"""
Run:  python code/explore.py
"""

import csv
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "dataset")

csv.field_size_limit(10_000_000)


# helpers


def hr(title, char="="):
    print("\n" + char * 78)
    print(title)
    print(char * 78)


def load(name):
    """Read dataset/<name> as a list of OrderedDicts, preserving raw strings."""
    path = os.path.join(DATASET, name)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [OrderedDict(r) for r in reader]
        cols = list(reader.fieldnames or [])
    return rows, cols


def col(rows, name):
    return [r.get(name) for r in rows]


def infer_dtype(values):
    """Crude pandas-like dtype inference over the non-blank values."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return "all-blank(object)"
    def is_int(v):
        try:
            int(v)
            return True
        except (TypeError, ValueError):
            return False
    def is_float(v):
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False
    if all(is_int(v) for v in vals):
        return "int64" if len(vals) == len(values) else "float64(has-blanks)"
    if all(is_float(v) for v in vals):
        return "float64"
    return "object(str)"


def describe(name):
    rows, cols = load(name)
    hr("FILE: %s" % name)
    print("shape: (%d rows, %d cols)" % (len(rows), len(cols)))
    print("columns: %s" % cols)
    print("\n%-34s %-22s %8s %8s %8s" % ("column", "dtype", "nblank", "nNaN*", "nuniq"))
    print("-" * 84)
    for c in cols:
        vals = col(rows, c)
        nblank = sum(1 for v in vals if v == "")
        nnone = sum(1 for v in vals if v is None)  # short row -> missing field
        nnan = sum(1 for v in vals if isinstance(v, str) and v.strip().lower()
                   in ("nan", "na", "n/a", "null", "none"))
        nuniq = len({v for v in vals if v not in (None, "")})
        print("%-34s %-22s %8d %8d %8d" % (c, infer_dtype(vals), nblank, nnone + nnan, nuniq))
    print("\n* nNaN column = literal 'nan'/'null'/'NA' strings or truly absent fields.")
    print("  Everything else missing is an EMPTY STRING, not NaN. This matters:")
    print("  csv.DictReader gives '' where pandas would give NaN.")

    print("\nhead(3):")
    for r in rows[:3]:
        print("  {")
        for k, v in r.items():
            sv = v if v is not None else "<absent>"
            if isinstance(sv, str) and len(sv) > 110:
                sv = sv[:110] + "...<%d chars>" % len(v)
            if isinstance(sv, str):
                sv = sv.replace("\n", "\\n")
            print("      %-30s = %r" % (k, sv))
        print("  }")

    # low-cardinality value_counts
    for c in cols:
        vals = col(rows, c)
        uniq = {v for v in vals if v not in (None, "")}
        if 0 < len(uniq) <= 15 and len(rows) > 15:
            print("\nvalue_counts(%s)  [incl. blanks]:" % c)
            for k, n in Counter("<blank>" if v in (None, "") else v
                                for v in vals).most_common():
                print("    %-40s %d" % (k, n))

    # duplicate id check on the first column if it looks like an id
    if cols and cols[0].endswith("_id"):
        c = Counter(col(rows, cols[0]))
        dups = {k: n for k, n in c.items() if n > 1}
        print("\nduplicate %s values: %d%s" % (cols[0], len(dups),
              (" -> " + str(list(dups.items())[:10])) if dups else ""))
    return rows, cols



# load everything

FILES = ["messages.csv", "output.csv", "sample_messages.csv", "users.csv",
         "groups.csv", "group_members.csv", "business_accounts.csv",
         "user_business_history.csv", "message_history.csv",
         "message_events.csv", "images.csv", "voice_notes.csv",
         "daily_notification_summary.csv"]

hr("PHASE 0 - DATASET EXPLORATION", "#")
print("dataset dir: %s" % DATASET)
print("files on disk: %s" % sorted(f for f in os.listdir(DATASET) if f.endswith(".csv")))

D = {}
for f in FILES:
    rows, cols = describe(f)
    D[f] = rows

messages = D["messages.csv"]
history = D["message_history.csv"]
events = D["message_events.csv"]
users = D["users.csv"]
groups = D["groups.csv"]
gmembers = D["group_members.csv"]
biz = D["business_accounts.csv"]
ubh = D["user_business_history.csv"]
images = D["images.csv"]
vnotes = D["voice_notes.csv"]
dns = D["daily_notification_summary.csv"]
sample = D["sample_messages.csv"]
out_tpl = D["output.csv"]

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append((label, ok))
    print("\n[%s] %s" % ("PASS" if ok else "FAIL", label))
    if detail:
        print(detail)


# --------------------------------------------------------------------------
# CHECK 1 - media referential integrity
# --------------------------------------------------------------------------
hr("CHECK 1 - MEDIA REFERENTIAL INTEGRITY")

img_ids = {r["image_id"] for r in images}
vn_ids = {r["voice_note_id"] for r in vnotes}
print("images.csv key column   : 'image_id'      (NOT 'media_id')  n=%d" % len(img_ids))
print("voice_notes.csv key col : 'voice_note_id' (NOT 'media_id')  n=%d" % len(vn_ids))

for src_name, src in (("messages.csv", messages), ("message_history.csv", history)):
    refs = [(r["message_id"], r["media_type"], r["media_id"])
            for r in src if r.get("media_id")]
    print("\n%s: %d rows with a non-blank media_id" % (src_name, len(refs)))
    print("  media_type breakdown: %s"
          % Counter(t for _, t, _ in refs).most_common())
    orphans = [(m, t, mid) for m, t, mid in refs
               if mid not in img_ids and mid not in vn_ids]
    mismatch = [(m, t, mid) for m, t, mid in refs
                if (t == "image" and mid not in img_ids)
                or (t == "voice" and mid not in vn_ids)]
    check("%s: every media_id resolves to images.csv or voice_notes.csv" % src_name,
          not orphans,
          "  orphan media_ids: %d %s" % (len(orphans), orphans[:20]))
    check("%s: media_type agrees with which table the id lives in" % src_name,
          not mismatch,
          "  mismatched: %d %s" % (len(mismatch), mismatch[:20]))
    # rows with media_type but no media_id and vice versa
    a = [r["message_id"] for r in src if r.get("media_type") and not r.get("media_id")]
    b = [r["message_id"] for r in src if r.get("media_id") and not r.get("media_type")]
    print("  rows w/ media_type but blank media_id: %d %s" % (len(a), a[:10]))
    print("  rows w/ media_id but blank media_type: %d %s" % (len(b), b[:10]))

missing_files = []
for r in images + vnotes:
    p = os.path.join(DATASET, r["file_path"].replace("/", os.sep))
    if not os.path.isfile(p):
        missing_files.append(r["file_path"])
check("every file_path in images.csv/voice_notes.csv exists under dataset/media/",
      not missing_files,
      "  missing files: %d %s" % (len(missing_files), missing_files[:20]))

used_media = {r["media_id"] for r in messages + history if r.get("media_id")}
unused = (img_ids | vn_ids) - used_media
print("\nmedia rows never referenced by any message: %d %s"
      % (len(unused), sorted(unused)[:20]))


# --------------------------------------------------------------------------
# CHECK 2 - id type / format consistency + orphans
# --------------------------------------------------------------------------
hr("CHECK 2 - ID TYPE / FORMAT CONSISTENCY AND ORPHANS")


def idset(rows, c):
    return {r[c] for r in rows if r.get(c)}


def fmt(vals):
    """Describe id string format: dtype, length range, regex-ish shape."""
    vals = sorted(vals)
    if not vals:
        return "EMPTY SET"
    lens = {len(v) for v in vals}
    shapes = Counter(re.sub(r"\d", "#", v) for v in vals)
    return ("dtype=%s len=%s shapes=%s e.g. %s"
            % (infer_dtype(vals), sorted(lens),
               [s for s, _ in shapes.most_common(3)], vals[:2]))


def compare(label, left_name, left, right_name, right, expect="subset"):
    print("\n--- %s" % label)
    print("  %-34s n=%-5d %s" % (left_name, len(left), fmt(left)))
    print("  %-34s n=%-5d %s" % (right_name, len(right), fmt(right)))
    orph = sorted(left - right)
    print("  in %s but NOT in %s: %d %s"
          % (left_name, right_name, len(orph), orph[:15]))
    print("  in %s but NOT in %s: %d"
          % (right_name, left_name, len(right - left)))
    if expect == "subset":
        check("%s -> %s referential integrity" % (left_name, right_name),
              not orph)
    return orph


U = idset(users, "user_id")
compare("user_id everywhere", "messages.user_id", idset(messages, "user_id"),
        "users.user_id", U)
compare("user_id everywhere", "group_members.user_id", idset(gmembers, "user_id"),
        "users.user_id", U)
compare("user_id everywhere", "message_history.user_id", idset(history, "user_id"),
        "users.user_id", U)
compare("user_id everywhere", "message_events.user_id", idset(events, "user_id"),
        "users.user_id", U)
compare("user_id everywhere", "user_business_history.user_id", idset(ubh, "user_id"),
        "users.user_id", U)
compare("user_id everywhere", "daily_notification_summary.user_id",
        idset(dns, "user_id"), "users.user_id", U)

G = idset(groups, "group_id")
compare("group_id", "messages.group_id", idset(messages, "group_id"),
        "groups.group_id", G)
compare("group_id", "group_members.group_id", idset(gmembers, "group_id"),
        "groups.group_id", G)
compare("group_id", "message_history.group_id", idset(history, "group_id"),
        "groups.group_id", G)

B = idset(biz, "business_id")
compare("business_id", "messages.business_id", idset(messages, "business_id"),
        "business_accounts.business_id", B)
compare("business_id", "user_business_history.business_id",
        idset(ubh, "business_id"), "business_accounts.business_id", B)
compare("business_id", "message_history.business_id", idset(history, "business_id"),
        "business_accounts.business_id", B)

print("\n--- sender_user_id -> users.csv  (drives 'unknown sender' logic)")
snd_m = idset(messages, "sender_user_id")
snd_h = idset(history, "sender_user_id")
print("  messages.sender_user_id        n=%-5d %s" % (len(snd_m), fmt(snd_m)))
print("  message_history.sender_user_id n=%-5d %s" % (len(snd_h), fmt(snd_h)))
print("  users.user_id                  n=%-5d %s" % (len(U), fmt(U)))
unk_m = sorted(snd_m - U)
unk_h = sorted(snd_h - U)
print("  messages senders NOT in users.csv        : %d/%d  %s"
      % (len(unk_m), len(snd_m), unk_m[:20]))
print("  message_history senders NOT in users.csv : %d/%d  %s"
      % (len(unk_h), len(snd_h), unk_h[:20]))
nrow_unknown = sum(1 for r in messages
                   if r.get("sender_user_id") and r["sender_user_id"] not in U)
print("  -> %d of %d messages.csv ROWS have a sender absent from users.csv"
      % (nrow_unknown, len(messages)))
print("  NOTE: users.csv holds only RECIPIENTS. Senders live in a wider id space;")
print("        absence from users.csv is NOT an orphan, it is the normal case.")

# do sender ids appear as group members?
gm_pairs = {(r["group_id"], r["user_id"]) for r in gmembers}
gm_users = {r["user_id"] for r in gmembers}
print("  senders that appear in group_members.user_id: %d/%d"
      % (len(snd_m & gm_users), len(snd_m)))

compare("message_id linkage", "message_events.message_id",
        idset(events, "message_id"), "message_history.message_id",
        idset(history, "message_id"))

print("\n--- message_id namespaces")
print("  messages.csv        : %s" % fmt(idset(messages, "message_id")))
print("  message_history.csv : %s" % fmt(idset(history, "message_id")))
print("  sample_messages.csv : %s" % fmt(idset(sample, "message_id")))
ov = idset(messages, "message_id") & idset(history, "message_id")
check("messages.csv and message_history.csv message_id namespaces are DISJOINT",
      not ov, "  overlap: %d %s" % (len(ov), sorted(ov)[:10]))

hist_no_events = idset(history, "message_id") - idset(events, "message_id")
print("\n  history messages with NO row in message_events: %d/%d"
      % (len(hist_no_events), len(idset(history, "message_id"))))
bad_pair = [(r["user_id"], r["message_id"]) for r in events
            if (r["message_id"], r["user_id"]) not in
            {(h["message_id"], h["user_id"]) for h in history}]
print("  message_events rows whose (message_id,user_id) mismatches history: %d %s"
      % (len(bad_pair), bad_pair[:10]))

# group membership integrity for group messages
missing_mem = [r["message_id"] for r in messages
               if r.get("group_id") and (r["group_id"], r["user_id"]) not in gm_pairs]
check("every group message recipient has a group_members row",
      not missing_mem,
      "  missing membership rows: %d %s" % (len(missing_mem), missing_mem[:15]))

# conversation_type vs which fk is populated
print("\nconversation_type vs populated foreign keys (messages.csv):")
combo = Counter((r["conversation_type"],
                 "group_id" if r.get("group_id") else "-",
                 "business_id" if r.get("business_id") else "-",
                 "sender_user_id" if r.get("sender_user_id") else "-")
                for r in messages)
for k, n in combo.most_common():
    print("    %-12s group=%-9s biz=%-12s sender=%-15s %d" % (k[0], k[1], k[2], k[3], n))


# --------------------------------------------------------------------------
# CHECK 3 - messages.csv vs dataset/output.csv template
# --------------------------------------------------------------------------
hr("CHECK 3 - messages.csv VS dataset/output.csv TEMPLATE")

with open(os.path.join(DATASET, "output.csv"), encoding="utf-8") as fh:
    raw = fh.read()
print("output.csv raw line count (incl. header): %d" % len(raw.splitlines()))
print("output.csv parsed data rows             : %d" % len(out_tpl))
print("messages.csv parsed data rows           : %d" % len(messages))
print("\noutput.csv header + first 3 raw lines:")
for ln in raw.splitlines()[:4]:
    print("    %r" % ln)

out_ids = [r["message_id"] for r in out_tpl]
msg_ids = [r["message_id"] for r in messages]
print("\noutput.csv template: all non-message_id cells blank? %s"
      % all(v == "" for r in out_tpl for k, v in r.items() if k != "message_id"))
print("output.csv duplicate message_ids: %d"
      % (len(out_ids) - len(set(out_ids))))
print("messages.csv duplicate message_ids: %d"
      % (len(msg_ids) - len(set(msg_ids))))
check("output.csv template message_ids are a SUBSET of messages.csv ids",
      set(out_ids) <= set(msg_ids),
      "  in template but not in messages.csv: %s"
      % sorted(set(out_ids) - set(msg_ids))[:10])
check("output.csv template covers EVERY messages.csv id",
      set(out_ids) == set(msg_ids),
      "  in messages.csv but MISSING from template: %d"
      % len(set(msg_ids) - set(out_ids)))
print("  -> first 5 template ids: %s" % out_ids[:5])
print("  -> messages.csv ids not in template (first 5): %s"
      % sorted(set(msg_ids) - set(out_ids))[:5])
print("\nEmbedded-newline explanation:")
emb = [r["message_id"] for r in messages if "\n" in (r.get("message_text") or "")]
print("  messages.csv rows whose message_text contains a literal newline: %d %s"
      % (len(emb), emb[:10]))
with open(os.path.join(DATASET, "messages.csv"), encoding="utf-8") as fh:
    nlines = len(fh.read().splitlines())
print("  messages.csv raw line count=%d vs parsed rows=%d (delta from quoted "
      "newlines)" % (nlines, len(messages)))


# --------------------------------------------------------------------------
# CHECK 4 - timestamps + do-not-disturb window
# --------------------------------------------------------------------------
hr("CHECK 4 - TIMESTAMP FORMATS AND QUIET HOURS")

for nm, rows in (("messages.csv", messages), ("message_history.csv", history),
                 ("sample_messages.csv", sample)):
    ts = [r["created_at"] for r in rows if r.get("created_at")]
    shapes = Counter(re.sub(r"\d", "#", t) for t in ts)
    parsed, bad = [], []
    for t in ts:
        try:
            parsed.append(datetime.strptime(t, "%Y-%m-%d %H:%M"))
        except ValueError:
            bad.append(t)
    print("\n%s.created_at" % nm)
    print("  n=%d blank=%d" % (len(ts), len(rows) - len(ts)))
    print("  distinct string shapes: %s" % shapes.most_common())
    print("  parses with '%%Y-%%m-%%d %%H:%%M': %d ok, %d bad %s"
          % (len(parsed), len(bad), bad[:5]))
    print("  tz-aware? NO - no offset, no 'Z', no tzinfo. Naive local time.")
    if parsed:
        print("  min=%s  max=%s  span=%s"
              % (min(parsed), max(parsed), max(parsed) - min(parsed)))
        print("  hour-of-day histogram: %s"
              % sorted(Counter(p.hour for p in parsed).items()))

print("\nother date-ish columns:")
for nm, rows, c in (("groups.csv", groups, "created_at"),
                    ("group_members.csv", gmembers, "joined_at"),
                    ("user_business_history.csv", ubh, "last_activity_at"),
                    ("user_business_history.csv", ubh, "promotions_opted_out_at"),
                    ("user_business_history.csv", ubh, "last_reply_at"),
                    ("daily_notification_summary.csv", dns, "date")):
    vals = [r[c] for r in rows if r.get(c)]
    print("  %-32s %-24s n=%-5d blank=%-4d shapes=%s"
          % (nm, c, len(vals), len(rows) - len(vals),
             [s for s, _ in Counter(re.sub(r"\d", "#", v) for v in vals).most_common(3)]))

print("\nusers.csv quiet hours column is named 'do_not_disturb_window'"
      " (NOT 'quiet_hours').")
dnd = [r["do_not_disturb_window"] for r in users]
print("  raw distinct values (%d users):" % len(users))
for v, n in Counter(dnd).most_common():
    print("      %-20r x%d" % (v, n))
wrap = 0
for v in set(dnd):
    if not v:
        continue
    s, e = v.split("-")
    sh, sm = map(int, s.split(":"))
    eh, em = map(int, e.split(":"))
    if (sh, sm) > (eh, em):
        wrap += 1
nonwrap = sorted({v for v in dnd if v and not (
    tuple(map(int, v.split("-")[0].split(":")))
    > tuple(map(int, v.split("-")[1].split(":"))))})
print("  windows that WRAP past midnight (start > end): %d of %d distinct"
      % (wrap, len({v for v in dnd if v})))
print("  NON-wrapping windows (start <= end): %s" % nonwrap)
print("""
  Implementation note for "is timestamp T inside the DND window":
    format is 'HH:MM-HH:MM', local naive, no date, no timezone.
    MOST windows wrap midnight (start > end), so a naive `start <= t <= end`
    comparison is WRONG for them (it would match nothing). But the two
    '00:00-HH:MM' windows do NOT wrap, so you cannot just always assume
    wrap-around either -- you must branch on start > end.
    Correct form:
        t = T.time(); s, e = window
        inside = (t >= s or t < e) if s > e else (s <= t < e)
    Treat the end bound as exclusive, and handle a blank/absent window as
    "no DND" rather than crashing.""")


# --------------------------------------------------------------------------
# CHECK 5 - sample_messages.csv style reference
# --------------------------------------------------------------------------
hr("CHECK 5 - sample_messages.csv AS STYLE GROUND TRUTH")

_, scols = load("sample_messages.csv")
print("columns (%d): %s" % (len(scols), scols))
print("rows: %d" % len(sample))
print("\nextra columns vs messages.csv: %s"
      % [c for c in scols if c not in load("messages.csv")[1]])

print("\naction distribution:")
for k, n in Counter(r["action"] for r in sample).most_common():
    print("    %-16s %3d  (%.1f%%)" % (k, n, 100.0 * n / len(sample)))
print("\nmessage_type distribution:")
for k, n in Counter(r["message_type"] for r in sample).most_common():
    print("    %-16s %3d  (%.1f%%)" % (k, n, 100.0 * n / len(sample)))
print("\naction x message_type:")
for k, n in Counter((r["action"], r["message_type"]) for r in sample).most_common():
    print("    %-8s %-16s %d" % (k[0], k[1], n))
print("\nconversation_type x action:")
for k, n in Counter((r["conversation_type"], r["action"]) for r in sample).most_common():
    print("    %-10s %-8s %d" % (k[0], k[1], n))

conf = [float(r["confidence"]) for r in sample]
print("\nconfidence: n=%d min=%.2f max=%.2f mean=%.4f"
      % (len(conf), min(conf), max(conf), sum(conf) / len(conf)))
print("  decimal places used: %s"
      % sorted({len(r["confidence"].split(".")[1]) for r in sample}))
print("  distinct values (%d): %s" % (len(set(conf)), sorted(set(conf))))
print("  by action:")
for a in sorted({r["action"] for r in sample}):
    cs = [float(r["confidence"]) for r in sample if r["action"] == a]
    print("    %-8s n=%-3d min=%.2f max=%.2f mean=%.3f"
          % (a, len(cs), min(cs), max(cs), sum(cs) / len(cs)))

wl = [len(r["reason"].split()) for r in sample]
print("\nreason length in words: min=%d max=%d mean=%.1f median=%d"
      % (min(wl), max(wl), sum(wl) / len(wl), sorted(wl)[len(wl) // 2]))
print("reason length in chars: min=%d max=%d mean=%.1f"
      % (min(len(r["reason"]) for r in sample),
         max(len(r["reason"]) for r in sample),
         sum(len(r["reason"]) for r in sample) / len(sample)))
print("reasons ending with '.': %d/%d"
      % (sum(1 for r in sample if r["reason"].strip().endswith(".")), len(sample)))
print("reasons containing a newline: %d"
      % sum(1 for r in sample if "\n" in r["reason"]))
print("empty reasons: %d" % sum(1 for r in sample if not r["reason"].strip()))
print("distinct reason strings: %d of %d rows"
      % (len({r["reason"] for r in sample}), len(sample)))
print("\n8 verbatim reason examples (with their action/type):")
seen = set()
shown = 0
for r in sample:
    key = (r["action"], r["message_type"])
    if key in seen:
        continue
    seen.add(key)
    print("  [%s / %s] %r" % (r["action"], r["message_type"], r["reason"]))
    shown += 1
    if shown >= 8:
        break

print("\nevidence_message_ids format:")
ev_raw = [r["evidence_message_ids"] for r in sample]
print("  blank: %d, literal 'none': %d"
      % (sum(1 for v in ev_raw if v == ""), sum(1 for v in ev_raw if v == "none")))
seps = Counter()
for v in ev_raw:
    if v and v != "none":
        for ch in ";,| ":
            if ch in v:
                seps[ch] += 1
print("  separator chars observed in multi-id cells: %s" % dict(seps))
counts = Counter(len([x for x in v.split(";") if x.strip()])
                 for v in ev_raw if v and v != "none")
print("  ids per non-'none' cell: %s" % sorted(counts.items()))
HIST = idset(history, "message_id")
allev, badev = [], []
for r in sample:
    v = r["evidence_message_ids"]
    if not v or v == "none":
        continue
    for x in v.split(";"):
        x = x.strip()
        allev.append(x)
        if x not in HIST:
            badev.append((r["message_id"], x))
print("  total evidence ids cited: %d (distinct %d)" % (len(allev), len(set(allev))))
check("every sample evidence_message_id resolves into message_history.csv",
      not badev, "  unresolved: %d %s" % (len(badev), badev[:10]))
same_user = sum(1 for r in sample for x in
                (r["evidence_message_ids"] or "").split(";")
                if x.strip() and x.strip() != "none"
                and any(h["message_id"] == x.strip() and h["user_id"] == r["user_id"]
                        for h in history))
print("  evidence ids belonging to the SAME user_id as the sample row: %d/%d"
      % (same_user, len(allev)))
inmsgs = [x for x in allev if x in idset(messages, "message_id")]
print("  evidence ids that are actually messages.csv ids (must be 0): %d" % len(inmsgs))

ov = idset(sample, "message_id") & idset(messages, "message_id")
check("sample_messages.csv ids are DISJOINT from messages.csv ids",
      not ov, "  overlap: %d %s" % (len(ov), sorted(ov)[:10]))
print("  sample rows also carry the full input schema (user_id, media_id, ...),")
print("  so they can be used as a dry-run eval set with known answers.")
smedia = [(r["message_id"], r["media_type"], r["media_id"])
          for r in sample if r.get("media_id")]
print("  sample rows with media: %d %s" % (len(smedia), smedia[:10]))
sorph = [t for t in smedia if t[2] not in img_ids and t[2] not in vn_ids]
print("  sample media ids not in images/voice_notes: %d %s" % (len(sorph), sorph))


# --------------------------------------------------------------------------
# CHECK 6 - media inventory on disk
# --------------------------------------------------------------------------
hr("CHECK 6 - MEDIA INVENTORY ON DISK")

for sub in ("images", "audio"):
    d = os.path.join(DATASET, "media", sub)
    if not os.path.isdir(d):
        print("MISSING DIR: %s" % d)
        continue
    fs = sorted(os.listdir(d))
    tot = sum(os.path.getsize(os.path.join(d, f)) for f in fs)
    print("\ndataset/media/%s: %d files, %.2f MB total" % (sub, len(fs), tot / 1e6))
    print("  extensions: %s"
          % Counter(os.path.splitext(f)[1].lower() for f in fs).most_common())
    sizes = sorted(os.path.getsize(os.path.join(d, f)) for f in fs)
    print("  size bytes: min=%d median=%d max=%d" % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))
    print("  files: %s" % fs)

# mp3 duration: parse frame headers with stdlib only (no deps)
BR = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
SR = {0: 44100, 1: 48000, 2: 32000}
print("\naudio durations (stdlib mp3 header scan, CBR assumption):")
adir = os.path.join(DATASET, "media", "audio")
durs = []
for f in sorted(os.listdir(adir)) if os.path.isdir(adir) else []:
    p = os.path.join(adir, f)
    try:
        with open(p, "rb") as fh:
            data = fh.read()
        i = 0
        if data[:3] == b"ID3":
            sz = 0
            for b in data[6:10]:
                sz = (sz << 7) | (b & 0x7F)
            i = 10 + sz
        d = None
        while i < len(data) - 4:
            if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                br = BR[(data[i + 2] >> 4) & 0xF] * 1000
                sr = SR.get((data[i + 2] >> 2) & 0x3)
                if br and sr:
                    d = (len(data) - i) * 8.0 / br
                break
            i += 1
        if d:
            durs.append(d)
            print("  %-16s %6.1f s" % (f, d))
        else:
            print("  %-16s <could not parse frame header>" % f)
    except Exception as exc:  # noqa: BLE001 - inventory only
        print("  %-16s <error: %s>" % (f, exc))
if durs:
    print("  total audio: %.1f s (%.1f min), mean %.1f s"
          % (sum(durs), sum(durs) / 60, sum(durs) / len(durs)))


# --------------------------------------------------------------------------
# CHECK 7 - .env sanity  (NEVER print the value)
# --------------------------------------------------------------------------
hr("CHECK 7 - .env / GEMINI_API_KEY SANITY (value never printed)")

env_path = os.path.join(ROOT, ".env")
print(".env exists: %s" % os.path.isfile(env_path))
if os.path.isfile(env_path):
    with open(env_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    print("line count: %d" % len(lines))
    for ln in lines:
        if "=" in ln:
            k, _, v = ln.partition("=")
            print("  raw key repr before strip: %r  (has trailing space: %s)"
                  % (k, k != k.rstrip()))
            vs = v.strip()
            quoted = len(vs) >= 2 and vs[0] == vs[-1] and vs[0] in "'\""
            print("  stripped key: %r" % k.strip())
            print("  value is wrapped in a matched quote pair: %s (quote char %r)"
                  % (quoted, vs[0] if quoted else None))
            print("  value length incl. quotes: %d   excl. quotes: %d"
                  % (len(vs), len(vs.strip("'\""))))
            print("  -> python-dotenv strips BOTH the whitespace around the key")
            print("     and a matched surrounding quote pair, so the two quirks")
            print("     in this .env line are both handled by the library.")
        elif ln.strip():
            print("  non-assignment line: %r" % ln)

try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
    have = "GEMINI_API_KEY" in os.environ
    print("\npython-dotenv: INSTALLED")
    check("load_dotenv() exposes os.environ['GEMINI_API_KEY'] despite the "
          "space before '='", have,
          "  key present=%s  length=%d"
          % (have, len(os.environ.get("GEMINI_API_KEY", ""))))
    weird = [k for k in os.environ if k.strip() != k and "GEMINI" in k.upper()]
    print("  env var names containing GEMINI: %s"
          % [k for k in os.environ if "GEMINI" in k.upper()])
    print("  whitespace-polluted GEMINI var names: %s" % weird)
except ImportError:
    print("\npython-dotenv: NOT INSTALLED in this interpreter.")
    print("  Cannot empirically confirm load_dotenv() behaviour yet.")
    print("  Emulating python-dotenv's documented parse (it strips whitespace")
    print("  around the key and around an unquoted value):")
    parsed = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        parsed[k.strip()] = v.strip().strip("'\"")
    print("  keys that WOULD be set: %s" % sorted(parsed))
    ok = "GEMINI_API_KEY" in parsed and len(parsed["GEMINI_API_KEY"]) > 0
    check("emulated dotenv parse yields non-empty 'GEMINI_API_KEY'", ok,
          "  key name present=%s  value length=%d"
          % ("GEMINI_API_KEY" in parsed, len(parsed.get("GEMINI_API_KEY", ""))))
    print("  RE-RUN THIS SCRIPT after `pip install python-dotenv` to confirm"
          " empirically.")

with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as fh:
    gi = fh.read()
check(".env is listed in .gitignore", ".env" in gi,
      "  .gitignore entries: %s" % [g for g in gi.splitlines() if g.strip()])


# --------------------------------------------------------------------------
# QUOTA BUDGET
# --------------------------------------------------------------------------
hr("QUOTA BUDGET - how many Gemini calls does a naive design need?")

n_total = len(messages)
mt = Counter(r["media_type"] or "text" for r in messages)
n_img = mt.get("image", 0)
n_voice = mt.get("voice", 0)
n_text = mt.get("text", 0)
uniq_img = len({r["media_id"] for r in messages if r["media_type"] == "image"})
uniq_vn = len({r["media_id"] for r in messages if r["media_type"] == "voice"})
print("messages.csv rows                : %d" % n_total)
print("  text-only (blank media_type)   : %d" % n_text)
print("  image messages                 : %d  (distinct image ids: %d)" % (n_img, uniq_img))
print("  voice messages                 : %d  (distinct voice ids: %d)" % (n_voice, uniq_vn))
print("\nNaive one-call-per-message design:")
print("  text classification calls      : %d  (1 per row)" % n_total)
print("  image understanding calls      : %d  (1 per DISTINCT image, cached)" % uniq_img)
print("  voice calls to Gemini          : 0   (AGENTS.md: transcribe locally"
      " with faster-whisper)")
print("  TOTAL Gemini requests          : %d" % (n_total + uniq_img))
print("\nPlus a dry-run over sample_messages.csv (%d rows, %d with media): "
      "+%d calls" % (len(sample), len(smedia), len(sample)))
print("GRAND TOTAL for one full build+run: ~%d requests"
      % (n_total + uniq_img + len(sample)))
for rpm in (10, 15, 30):
    n = n_total + uniq_img
    print("  at %2d RPM the %d calls take ~%.1f min of wall clock"
          % (rpm, n, n / float(rpm)))

hr("SUMMARY OF CHECKS")
for label, ok in RESULTS:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("\n%d/%d checks passed." % (sum(1 for _, o in RESULTS if o), len(RESULTS)))
sys.stdout.flush()
