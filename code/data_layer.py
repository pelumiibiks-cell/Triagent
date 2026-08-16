from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from datetime import date, datetime, time
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("data_layer")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")

csv.field_size_limit(10_000_000)

DATETIME_FMT = "%Y-%m-%d %H:%M"   # confirmed in Phase 0: naive, no timezone
DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# value coercion
# ---------------------------------------------------------------------------

def _blank_to_none(v: Optional[str]) -> Optional[str]:
    """The dataset's only missing-value shape is the empty string."""
    if v is None:
        return None
    v = v.strip()
    return v or None


def _as_int(v: Optional[str]) -> Optional[int]:
    v = _blank_to_none(v)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        log.warning("expected int, got %r - storing None", v)
        return None


def _as_float(v: Optional[str]) -> Optional[float]:
    v = _blank_to_none(v)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        log.warning("expected float, got %r - storing None", v)
        return None


def _as_bool(v: Optional[str]) -> Optional[bool]:
    """0/1 flag columns -> bool. Anything unexpected -> None + warning."""
    v = _blank_to_none(v)
    if v is None:
        return None
    if v in ("0", "1"):
        return v == "1"
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    log.warning("expected 0/1 flag, got %r - storing None", v)
    return None


def _as_datetime(v: Optional[str]) -> Optional[datetime]:
    v = _blank_to_none(v)
    if v is None:
        return None
    try:
        return datetime.strptime(v, DATETIME_FMT)
    except ValueError:
        log.warning("unparseable datetime %r - storing None", v)
        return None


def _as_date(v: Optional[str]) -> Optional[date]:
    v = _blank_to_none(v)
    if v is None:
        return None
    try:
        return datetime.strptime(v, DATE_FMT).date()
    except ValueError:
        log.warning("unparseable date %r - storing None", v)
        return None


# Per-file column coercions. Any column not listed stays a str-or-None.
SCHEMA: Dict[str, Dict[str, Any]] = {
    "messages.csv": {
        "created_at": _as_datetime,
        "forwarded_count": _as_int,
    },
    "sample_messages.csv": {
        "created_at": _as_datetime,
        "forwarded_count": _as_int,
        "confidence": _as_float,
    },
    "output.csv": {
        "confidence": _as_float,
    },
    "users.csv": {
        "messages_opened_30d": _as_int,
        "messages_replied_30d": _as_int,
        "notifications_dismissed_30d": _as_int,
        "messages_reported_30d": _as_int,
    },
    "groups.csv": {
        "member_count": _as_int,
        "admin_count": _as_int,
        "created_at": _as_date,
        "messages_30d": _as_int,
    },
    "group_members.csv": {
        "joined_at": _as_date,
        "messages_sent_30d": _as_int,
        "messages_read_30d": _as_int,
        "replies_sent_30d": _as_int,
        "notifications_dismissed_30d": _as_int,
        "group_muted_by_user": _as_bool,
    },
    "business_accounts.csv": {
        "verified": _as_bool,
        "account_age_days": _as_int,
        "messages_sent_30d": _as_int,
        "user_reports_30d": _as_int,
        "domain_used_by_sender_age_days": _as_int,
    },
    "user_business_history.csv": {
        "last_activity_at": _as_datetime,
        "allows_promotions": _as_bool,
        "promotions_opted_out_at": _as_datetime,
        "activity_count_180d": _as_int,
        "messages_opened_30d": _as_int,
        "messages_dismissed_30d": _as_int,
        "messages_replied_30d": _as_int,
        "last_reply_at": _as_datetime,
    },
    "message_history.csv": {
        "created_at": _as_datetime,
        "forwarded_count": _as_int,
    },
    "message_events.csv": {
        "message_opened": _as_bool,
        "message_replied": _as_bool,
        "reaction_time_minutes": _as_int,
        "notification_dismissed": _as_bool,
        "muted_after_message": _as_bool,
        "message_reported": _as_bool,
    },
    "images.csv": {},
    "voice_notes.csv": {},
    "daily_notification_summary.csv": {
        "date": _as_date,
        "notifications_sent": _as_int,
        "notifications_dismissed": _as_int,
    },
}

ALL_FILES = list(SCHEMA.keys())


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class Record:
    """One CSV row: coerced values, plus the verbatim strings on `.raw`.

    Mapping-style access (`rec["col"]`) returns the coerced value and raises
    KeyError only for a column that does not exist in the file at all -- a
    genuine programming error. A column that exists but is blank yields None.
    """

    __slots__ = ("_d", "raw", "_source")

    def __init__(self, values: Dict[str, Any], raw: Dict[str, str], source: str):
        self._d = values
        self.raw = raw
        self._source = source

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str, default: Any = None) -> Any:
        v = self._d.get(key, default)
        return default if v is None else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self.__getattribute__("_d")[key]
        except KeyError:
            raise AttributeError(
                "%s has no column %r (columns: %s)"
                % (self._source, key, sorted(self._d))
            ) from None

    def keys(self) -> Iterable[str]:
        return self._d.keys()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._d)

    def __repr__(self) -> str:
        return "<Record %s %s>" % (self._source, self._d)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class DatasetError(RuntimeError):
    """Raised loudly at load time when the dataset itself is unusable."""


def _read_csv(filename: str, dataset_dir: str) -> List[Record]:
    path = os.path.join(dataset_dir, filename)
    if not os.path.isfile(path):
        raise DatasetError(
            "Required dataset file is missing: %s\n"
            "Expected it under %s. The dataset/ directory must contain all %d "
            "CSVs (%s)." % (path, dataset_dir, len(ALL_FILES), ", ".join(ALL_FILES))
        )
    coercions = SCHEMA[filename]
    out: List[Record] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise DatasetError("Dataset file has no header row: %s" % path)
        unknown = set(coercions) - set(reader.fieldnames)
        if unknown:
            raise DatasetError(
                "%s: schema expects column(s) %s that the file does not have. "
                "Header is: %s" % (filename, sorted(unknown), reader.fieldnames)
            )
        for raw_row in reader:
            raw = {k: (v if v is not None else "") for k, v in raw_row.items()}
            values: Dict[str, Any] = {}
            for col, sval in raw.items():
                fn = coercions.get(col)
                values[col] = fn(sval) if fn else _blank_to_none(sval)
            out.append(Record(values, raw, filename))
    return out


# ---------------------------------------------------------------------------
# sender keys (Phase 4 needs to retrieve history by "same sender")
# ---------------------------------------------------------------------------

def sender_keys(row: Record) -> List[str]:
    """All sender-ish keys a message can be retrieved by.

    A group message carries BOTH a group_id and a sender_user_id, so it is
    indexed under both -- Phase 4 may want "same group" or "same person".
    Returns [] when nothing identifies the sender.
    """
    keys: List[str] = []
    if row.get("group_id"):
        keys.append("group:%s" % row["group_id"])
    if row.get("business_id"):
        keys.append("business:%s" % row["business_id"])
    if row.get("sender_user_id"):
        keys.append("user:%s" % row["sender_user_id"])
    return keys


def primary_sender_key(row: Record) -> Optional[str]:
    """The single most specific sender key: business > group > user."""
    if row.get("business_id"):
        return "business:%s" % row["business_id"]
    if row.get("group_id"):
        return "group:%s" % row["group_id"]
    if row.get("sender_user_id"):
        return "user:%s" % row["sender_user_id"]
    return None


# ---------------------------------------------------------------------------
# do-not-disturb
# ---------------------------------------------------------------------------

def parse_dnd_window(window: Optional[str]) -> Optional[Tuple[time, time]]:
    """'HH:MM-HH:MM' -> (start, end). None/blank/malformed -> None (no DND)."""
    window = _blank_to_none(window)
    if window is None or "-" not in window:
        if window is not None:
            log.warning("malformed do_not_disturb_window %r - treating as no DND", window)
        return None
    s_txt, _, e_txt = window.partition("-")
    try:
        sh, sm = (int(x) for x in s_txt.strip().split(":"))
        eh, em = (int(x) for x in e_txt.strip().split(":"))
        return time(sh, sm), time(eh, em)
    except ValueError:
        log.warning("malformed do_not_disturb_window %r - treating as no DND", window)
        return None


def in_dnd_window(window: Optional[str], when: Optional[datetime]) -> bool:
    """Is `when` inside the DND window?

    Phase 0 finding: 12 of the 14 distinct windows wrap past midnight
    (start > end), but '00:00-06:30' and '00:00-07:00' do NOT. So the
    comparison must branch rather than always assuming wrap-around.
    End bound is exclusive. A missing window means "no DND" -> False.
    """
    parsed = parse_dnd_window(window)
    if parsed is None or when is None:
        return False
    start, end = parsed
    t = when.time()
    if start > end:                       # wraps midnight
        return t >= start or t < end
    return start <= t < end               # same-day window


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------

class Dataset:
    """All 13 CSVs, loaded and indexed. Shared context for Phases 2-7."""

    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.media_dir = os.path.join(dataset_dir, "media")

        # --- raw tables, source order preserved -----------------------------
        self.messages: List[Record] = _read_csv("messages.csv", dataset_dir)
        self.sample_messages: List[Record] = _read_csv("sample_messages.csv", dataset_dir)
        self.output_template: List[Record] = _read_csv("output.csv", dataset_dir)
        self._users: List[Record] = _read_csv("users.csv", dataset_dir)
        self._groups: List[Record] = _read_csv("groups.csv", dataset_dir)
        self._group_members: List[Record] = _read_csv("group_members.csv", dataset_dir)
        self._businesses: List[Record] = _read_csv("business_accounts.csv", dataset_dir)
        self._user_business: List[Record] = _read_csv("user_business_history.csv", dataset_dir)
        self.history_rows: List[Record] = _read_csv("message_history.csv", dataset_dir)
        self._events: List[Record] = _read_csv("message_events.csv", dataset_dir)
        self._images: List[Record] = _read_csv("images.csv", dataset_dir)
        self._voice_notes: List[Record] = _read_csv("voice_notes.csv", dataset_dir)
        self._daily: List[Record] = _read_csv("daily_notification_summary.csv", dataset_dir)

        # --- indexes --------------------------------------------------------
        self.users: Dict[str, Record] = self._index(self._users, "user_id")
        self.groups: Dict[str, Record] = self._index(self._groups, "group_id")
        self.businesses: Dict[str, Record] = self._index(self._businesses, "business_id")
        self.images: Dict[str, Record] = self._index(self._images, "image_id")
        self.voice_notes: Dict[str, Record] = self._index(self._voice_notes, "voice_note_id")
        self.history: Dict[str, Record] = self._index(self.history_rows, "message_id")
        self.events: Dict[str, Record] = self._index(self._events, "message_id")

        self.group_members: Dict[Tuple[str, str], Record] = {}
        self.members_by_group: Dict[str, List[Record]] = defaultdict(list)
        self.groups_by_user: Dict[str, List[Record]] = defaultdict(list)
        for r in self._group_members:
            gid, uid = r["group_id"], r["user_id"]
            if gid is None or uid is None:
                log.warning("group_members row with blank key: %r", r.raw)
                continue
            self.group_members[(gid, uid)] = r
            self.members_by_group[gid].append(r)
            self.groups_by_user[uid].append(r)

        self.user_business: Dict[Tuple[str, str], Record] = {}
        self.businesses_by_user: Dict[str, List[Record]] = defaultdict(list)
        for r in self._user_business:
            uid, bid = r["user_id"], r["business_id"]
            if uid is None or bid is None:
                log.warning("user_business_history row with blank key: %r", r.raw)
                continue
            self.user_business[(uid, bid)] = r
            self.businesses_by_user[uid].append(r)

        self.daily_summary: Dict[Tuple[str, date], Record] = {}
        self.daily_by_user: Dict[str, List[Record]] = defaultdict(list)
        for r in self._daily:
            uid, d = r["user_id"], r["date"]
            if uid is None or d is None:
                log.warning("daily_notification_summary row with blank key: %r", r.raw)
                continue
            self.daily_summary[(uid, d)] = r
            self.daily_by_user[uid].append(r)

        # history indexes, newest-first (Phase 4 ranks by recency)
        self.history_by_user: Dict[str, List[Record]] = defaultdict(list)
        self.history_by_user_sender: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
        for r in self.history_rows:
            uid = r["user_id"]
            if uid is None:
                log.warning("message_history row with blank user_id: %r", r.raw)
                continue
            self.history_by_user[uid].append(r)
            for sk in sender_keys(r):
                self.history_by_user_sender[(uid, sk)].append(r)

        def _recency(rec: Record) -> Tuple[datetime, str]:
            # Phase 0: history timestamps cluster on 3 hours, so ties are
            # common -- break them deterministically by message_id.
            return (rec["created_at"] or datetime.min, rec["message_id"] or "")

        for bucket in self.history_by_user.values():
            bucket.sort(key=_recency, reverse=True)
        for bucket in self.history_by_user_sender.values():
            bucket.sort(key=_recency, reverse=True)

        # freeze the defaultdicts so a missing key can't silently create one
        self.members_by_group = dict(self.members_by_group)
        self.groups_by_user = dict(self.groups_by_user)
        self.businesses_by_user = dict(self.businesses_by_user)
        self.daily_by_user = dict(self.daily_by_user)
        self.history_by_user = dict(self.history_by_user)
        self.history_by_user_sender = dict(self.history_by_user_sender)

        self._media_warned: set = set()

    @staticmethod
    def _index(rows: List[Record], key: str) -> Dict[str, Record]:
        out: Dict[str, Record] = {}
        for r in rows:
            k = r[key]
            if k is None:
                log.warning("row with blank %s skipped: %r", key, r.raw)
                continue
            if k in out:
                log.warning("duplicate %s=%r - keeping the first row", key, k)
                continue
            out[k] = r
        return out

    # -- None-safe accessors -------------------------------------------------
    # Every one takes a possibly-None foreign key and returns None, never
    # raising. Phase 0 confirmed group_id/business_id/sender_user_id are blank
    # on ~43%/73%/27% of messages.csv rows respectively.

    def get_user(self, user_id: Optional[str]) -> Optional[Record]:
        return self.users.get(user_id) if user_id else None

    def get_group(self, group_id: Optional[str]) -> Optional[Record]:
        return self.groups.get(group_id) if group_id else None

    def get_business(self, business_id: Optional[str]) -> Optional[Record]:
        return self.businesses.get(business_id) if business_id else None

    def get_member(self, group_id: Optional[str],
                   user_id: Optional[str]) -> Optional[Record]:
        if not group_id or not user_id:
            return None
        return self.group_members.get((group_id, user_id))

    def get_user_business(self, user_id: Optional[str],
                          business_id: Optional[str]) -> Optional[Record]:
        if not user_id or not business_id:
            return None
        return self.user_business.get((user_id, business_id))

    def get_history(self, message_id: Optional[str]) -> Optional[Record]:
        return self.history.get(message_id) if message_id else None

    def get_event(self, message_id: Optional[str]) -> Optional[Record]:
        return self.events.get(message_id) if message_id else None

    def get_daily_summary(self, user_id: Optional[str],
                          day: Optional[date]) -> Optional[Record]:
        if not user_id or day is None:
            return None
        return self.daily_summary.get((user_id, day))

    def get_members_of_group(self, group_id: Optional[str]) -> List[Record]:
        return self.members_by_group.get(group_id, []) if group_id else []

    def get_history_for_user(self, user_id: Optional[str]) -> List[Record]:
        return self.history_by_user.get(user_id, []) if user_id else []

    def get_history_for_user_sender(self, user_id: Optional[str],
                                    sender_key: Optional[str]) -> List[Record]:
        if not user_id or not sender_key:
            return []
        return self.history_by_user_sender.get((user_id, sender_key), [])

    # -- media ---------------------------------------------------------------

    def media_path(self, media_id: Optional[str],
                   media_type: Optional[str] = None) -> Optional[str]:
        """Absolute path for a media id, or None (never raises).

        Phase 0 showed all 33 media rows resolve today, but a missing row or a
        missing file on disk must degrade to metadata-only classification
        rather than kill the run. Warns once per distinct id.
        """
        if not media_id:
            return None

        rec = None
        if media_type == "image":
            rec = self.images.get(media_id)
        elif media_type == "voice":
            rec = self.voice_notes.get(media_id)
        if rec is None:                       # unknown/absent media_type: try both
            rec = self.images.get(media_id) or self.voice_notes.get(media_id)

        if rec is None:
            self._warn_media(media_id, "media_id %r not found in images.csv or "
                                       "voice_notes.csv" % media_id)
            return None

        rel = rec["file_path"]
        if not rel:
            self._warn_media(media_id, "media_id %r has a blank file_path" % media_id)
            return None

        path = os.path.join(self.dataset_dir, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            self._warn_media(media_id, "media file for %r is missing on disk: %s"
                             % (media_id, path))
            return None
        return path

    def _warn_media(self, media_id: str, msg: str) -> None:
        if media_id not in self._media_warned:
            self._media_warned.add(media_id)
            log.warning("%s - degrading to metadata-only", msg)

    # -- do not disturb ------------------------------------------------------

    def user_in_dnd(self, user_id: Optional[str],
                    when: Optional[datetime]) -> bool:
        """Is `when` inside this user's DND window? Unknown user -> False."""
        user = self.get_user(user_id)
        if user is None:
            return False
        return in_dnd_window(user["do_not_disturb_window"], when)

    def __repr__(self) -> str:
        return ("<Dataset messages=%d history=%d users=%d groups=%d businesses=%d>"
                % (len(self.messages), len(self.history_rows), len(self.users),
                   len(self.groups), len(self.businesses)))


def load_dataset(dataset_dir: str = DATASET_DIR) -> Dataset:
    """Load and index the full dataset. Raises DatasetError if a file is missing."""
    if not os.path.isdir(dataset_dir):
        raise DatasetError(
            "Dataset directory not found: %s\nRun from the repo root, or pass "
            "dataset_dir explicitly." % dataset_dir
        )
    return Dataset(dataset_dir)


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _self_check() -> int:
    logging.basicConfig(level=logging.WARNING, format="  [warn] %(message)s")

    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
        if detail:
            print("         %s" % detail)
        if not ok:
            failures.append(label)

    print("=" * 74)
    print("PHASE 1 SELF-CHECK - data_layer.py")
    print("=" * 74)
    ds = load_dataset()
    print("loaded: %r\n" % ds)

    # -- 1. row counts vs Phase 0 -------------------------------------------
    print("1. ROW COUNTS vs Phase 0")
    expected = [
        ("messages.csv", len(ds.messages), 110),
        ("output.csv", len(ds.output_template), 110),
        ("sample_messages.csv", len(ds.sample_messages), 30),
        ("users.csv", len(ds.users), 54),
        ("groups.csv", len(ds.groups), 23),
        ("group_members.csv", len(ds.group_members), 401),
        ("business_accounts.csv", len(ds.businesses), 110),
        ("user_business_history.csv", len(ds.user_business), 106),
        ("message_history.csv", len(ds.history), 412),
        ("message_events.csv", len(ds.events), 412),
        ("images.csv", len(ds.images), 20),
        ("voice_notes.csv", len(ds.voice_notes), 13),
        ("daily_notification_summary.csv", len(ds.daily_summary), 756),
    ]
    for name, got, want in expected:
        check("%-32s %4d == %d" % (name, got, want), got == want)

    # -- 2. orphan foreign keys across the 18 joins --------------------------
    print("\n2. ORPHAN FOREIGN KEYS (18 joins)")
    joins = [
        ("messages.user_id -> users", {m["user_id"] for m in ds.messages}, set(ds.users)),
        ("group_members.user_id -> users",
         {u for _, u in ds.group_members}, set(ds.users)),
        ("message_history.user_id -> users",
         {h["user_id"] for h in ds.history_rows}, set(ds.users)),
        ("message_events.user_id -> users",
         {e["user_id"] for e in ds.events.values()}, set(ds.users)),
        ("user_business_history.user_id -> users",
         {u for u, _ in ds.user_business}, set(ds.users)),
        ("daily_summary.user_id -> users",
         {u for u, _ in ds.daily_summary}, set(ds.users)),
        ("messages.group_id -> groups",
         {m["group_id"] for m in ds.messages}, set(ds.groups)),
        ("group_members.group_id -> groups",
         {g for g, _ in ds.group_members}, set(ds.groups)),
        ("message_history.group_id -> groups",
         {h["group_id"] for h in ds.history_rows}, set(ds.groups)),
        ("messages.business_id -> businesses",
         {m["business_id"] for m in ds.messages}, set(ds.businesses)),
        ("user_business_history.business_id -> businesses",
         {b for _, b in ds.user_business}, set(ds.businesses)),
        ("message_history.business_id -> businesses",
         {h["business_id"] for h in ds.history_rows}, set(ds.businesses)),
        ("messages.sender_user_id -> users",
         {m["sender_user_id"] for m in ds.messages}, set(ds.users)),
        ("message_history.sender_user_id -> users",
         {h["sender_user_id"] for h in ds.history_rows}, set(ds.users)),
        ("message_events.message_id -> message_history",
         set(ds.events), set(ds.history)),
        ("message_history.message_id -> message_events",
         set(ds.history), set(ds.events)),
        ("output.csv ids -> messages.csv ids",
         {o["message_id"] for o in ds.output_template},
         {m["message_id"] for m in ds.messages}),
        ("sample evidence ids -> message_history",
         {x.strip() for s in ds.sample_messages
          for x in (s["evidence_message_ids"] or "none").split(";")
          if x.strip() and x.strip() != "none"},
         set(ds.history)),
    ]
    for label, left, right in joins:
        left = {v for v in left if v is not None}   # blank FK is not an orphan
        orphans = sorted(left - right)
        check("%-48s orphans=%d" % (label, len(orphans)), not orphans,
              str(orphans[:8]) if orphans else "")
    print("   (%d joins checked)" % len(joins))

    # -- 3. every media_id in messages.csv resolves to a real file -----------
    print("\n3. MEDIA RESOLUTION (messages.csv)")
    media_rows = [m for m in ds.messages if m["media_id"]]
    resolved, unresolved = [], []
    for m in media_rows:
        p = ds.media_path(m["media_id"], m["media_type"])
        (resolved if p else unresolved).append(m["media_id"])
    check("messages.csv media_ids resolve to existing files: %d/%d"
          % (len(resolved), len(media_rows)),
          len(media_rows) == 23 and not unresolved,
          "unresolved: %s" % unresolved[:8] if unresolved else "")
    kinds: Dict[str, int] = {}
    for m in media_rows:
        kinds[m["media_type"]] = kinds.get(m["media_type"], 0) + 1
    print("         by media_type: %s" % sorted(kinds.items()))
    check("unknown media_id degrades to None (no raise)",
          ds.media_path("img_does_not_exist", "image") is None)
    check("blank/None media_id returns None (no raise)",
          ds.media_path(None) is None and ds.media_path("") is None)

    # -- 4. personal message with blank group_id/business_id -----------------
    print("\n4. BLANK FOREIGN KEYS DEGRADE, NEVER RAISE")
    personal = [m for m in ds.messages
                if m["conversation_type"] == "personal"
                and m["group_id"] is None and m["business_id"] is None]
    check("found personal messages with blank group_id AND business_id: %d"
          % len(personal), bool(personal))
    pm = personal[0]
    print("         sample: message_id=%s user_id=%s sender_user_id=%s"
          % (pm["message_id"], pm["user_id"], pm["sender_user_id"]))
    print("         raw group_id=%r -> coerced %r"
          % (pm.raw["group_id"], pm["group_id"]))
    check("get_group(None) -> None", ds.get_group(pm["group_id"]) is None)
    check("get_business(None) -> None", ds.get_business(pm["business_id"]) is None)
    check("get_member(None, user) -> None",
          ds.get_member(pm["group_id"], pm["user_id"]) is None)
    check("get_user_business(user, None) -> None",
          ds.get_user_business(pm["user_id"], pm["business_id"]) is None)
    check("get_members_of_group(None) -> []",
          ds.get_members_of_group(pm["group_id"]) == [])
    check("get_user(sender) resolves for this personal message",
          ds.get_user(pm["sender_user_id"]) is not None)
    check("get_history_for_user_sender(user, primary_sender_key) is a list",
          isinstance(ds.get_history_for_user_sender(
              pm["user_id"], primary_sender_key(pm)), list))
    check("get_user('u_does_not_exist') -> None",
          ds.get_user("u_does_not_exist") is None)
    check("blank optional field is None, not '' (promotions_opted_out_at)",
          any(r["promotions_opted_out_at"] is None
              for r in ds.user_business.values()))
    check("blank official_domain is None, not ''",
          all(b["official_domain"] is None or b["official_domain"].strip()
              for b in ds.businesses.values()))

    # -- 5. DND helper, wrap and non-wrap ------------------------------------
    print("\n5. DO-NOT-DISTURB WINDOW (wrap and non-wrap)")

    def at(hh: int, mm: int) -> datetime:
        return datetime(2026, 7, 30, hh, mm)

    wrap = "22:00-07:00"       # 12 of 14 windows look like this
    cases_wrap = [
        (at(22, 0), True, "start bound, inclusive"),
        (at(23, 30), True, "before midnight"),
        (at(2, 0), True, "after midnight"),
        (at(6, 59), True, "just before end"),
        (at(7, 0), False, "end bound, exclusive"),
        (at(12, 0), False, "midday"),
        (at(21, 59), False, "just before start"),
    ]
    for when, want, note in cases_wrap:
        got = in_dnd_window(wrap, when)
        check("%s @ %s -> %-5s (%s)" % (wrap, when.strftime("%H:%M"), got, note),
              got == want)

    nonwrap = "00:00-06:30"    # the 2 windows that do NOT wrap
    cases_non = [
        (at(0, 0), True, "start bound, inclusive"),
        (at(3, 0), True, "inside"),
        (at(6, 29), True, "just before end"),
        (at(6, 30), False, "end bound, exclusive"),
        (at(12, 0), False, "midday"),
        (at(23, 0), False, "late evening - would be a false positive if the "
                           "code always assumed wrap-around"),
    ]
    for when, want, note in cases_non:
        got = in_dnd_window(nonwrap, when)
        check("%s @ %s -> %-5s (%s)" % (nonwrap, when.strftime("%H:%M"), got, note),
              got == want)

    check("blank window -> no DND", in_dnd_window(None, at(3, 0)) is False
          and in_dnd_window("", at(3, 0)) is False)
    check("malformed window -> no DND (warns)",
          in_dnd_window("not-a-window", at(3, 0)) is False)
    check("None timestamp -> no DND", in_dnd_window(wrap, None) is False)
    check("user_in_dnd on unknown user -> False",
          ds.user_in_dnd("u_does_not_exist", at(3, 0)) is False)

    # every real window in users.csv must parse
    bad = [u["do_not_disturb_window"] for u in ds.users.values()
           if parse_dnd_window(u["do_not_disturb_window"]) is None]
    check("all %d users' DND windows parse: %d bad" % (len(ds.users), len(bad)),
          not bad, str(bad[:5]))
    n_wrap = sum(1 for w in {u["do_not_disturb_window"] for u in ds.users.values()}
                 if (lambda p: p and p[0] > p[1])(parse_dnd_window(w)))
    distinct = len({u["do_not_disturb_window"] for u in ds.users.values()})
    print("         %d of %d distinct windows wrap midnight (Phase 0: 12 of 14)"
          % (n_wrap, distinct))
    in_dnd_now = sum(1 for m in ds.messages
                     if ds.user_in_dnd(m["user_id"], m["created_at"]))
    print("         messages.csv rows arriving inside their recipient's DND: %d/%d"
          % (in_dnd_now, len(ds.messages)))

    # -- 6. coercion + index spot-checks -------------------------------------
    print("\n6. COERCION AND INDEX SPOT-CHECKS")
    check("created_at coerced to datetime",
          all(isinstance(m["created_at"], datetime) for m in ds.messages))
    check("forwarded_count coerced to int",
          all(isinstance(m["forwarded_count"], int) for m in ds.messages))
    check("group_muted_by_user coerced to bool",
          all(isinstance(r["group_muted_by_user"], bool)
              for r in ds.group_members.values()))
    check("business.verified coerced to bool",
          all(isinstance(b["verified"], bool) for b in ds.businesses.values()))
    check("groups.created_at coerced to date (not datetime)",
          all(isinstance(g["created_at"], date)
              and not isinstance(g["created_at"], datetime)
              for g in ds.groups.values()))
    check("sample confidence coerced to float in [0,1]",
          all(isinstance(s["confidence"], float) and 0.0 <= s["confidence"] <= 1.0
              for s in ds.sample_messages))
    check("no coerced value is an empty string",
          not [1 for t in (ds.messages, ds.history_rows, ds.sample_messages)
               for r in t for v in r.as_dict().values() if v == ""])
    check(".raw preserves verbatim source strings",
          all(isinstance(v, str) for v in ds.messages[0].raw.values()))
    ex = next(m for m in ds.messages if m["group_id"])
    print("         raw created_at=%r -> %r"
          % (ex.raw["created_at"], ex["created_at"]))

    tot_hist = sum(len(v) for v in ds.history_by_user.values())
    check("history_by_user covers every history row: %d == %d"
          % (tot_hist, len(ds.history_rows)), tot_hist == len(ds.history_rows))
    check("history_by_user buckets sorted newest-first",
          all(all(b[i]["created_at"] >= b[i + 1]["created_at"]
                  for i in range(len(b) - 1))
              for b in ds.history_by_user.values()))
    check("history_by_user_sender keys are non-empty: %d"
          % len(ds.history_by_user_sender), bool(ds.history_by_user_sender))
    gm = next(m for m in ds.messages if m["group_id"])
    print("         sender_keys(group msg %s) = %s" % (gm["message_id"], sender_keys(gm)))
    print("         primary_sender_key = %r" % primary_sender_key(gm))
    print("         history for (%s, %s): %d rows"
          % (gm["user_id"], primary_sender_key(gm),
             len(ds.get_history_for_user_sender(gm["user_id"],
                                                primary_sender_key(gm)))))
    check("missing history bucket returns [] not KeyError",
          ds.get_history_for_user_sender("u_001", "group:group_999") == [])
    check("every history row has a paired event",
          all(ds.get_event(h["message_id"]) is not None for h in ds.history_rows))
    check("Record.get() on a blank column returns the default",
          pm.get("group_id", "FALLBACK") == "FALLBACK")

    # -- summary -------------------------------------------------------------
    print("\n" + "=" * 74)
    if failures:
        print("SELF-CHECK FAILED - %d failing check(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SELF-CHECK PASSED - all checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
