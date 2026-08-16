"""Phase 2 - Per-message context builder.

"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer import (  # noqa: E402
    Dataset,
    Record,
    load_dataset,
    primary_sender_key,
    sender_keys,
)


# users.csv has exactly six columns  user_id, do_not_disturb_window, and four
# 30-day counters. There is NO display name, handle, phone, or alias column
# anywhere in the dataset for a *user*. groups.csv has a `group_name`, but no
# message text contains its group's name (checked: 0 of 140). So name-based
# mention matching is NOT implementable here, and pretending otherwise would be
# fabricating a capability the dataset cannot support.
#
# What IS reliably detectable: the corpus writes mentions as the literal user
# id prefixed with '@', e.g. "@u_007". Across messages.csv + sample_messages.csv
# there are 7 such messages, every one a group message, and in every case the
# mentioned id is the recipient's own user_id.
#
# So `direct_mention` is defined as: an explicit "@<user_id>" token matching the
# recipient. That is a high-precision signal, which is what Phase 6 needs -- it
# is the condition allowing `notify` to override a muted group, so a false
# positive there would defeat the mute.
#
# Second-person address ("can you send", "pls confirm") is far too common in
# group chat to be a reliable *direct* mention, so it is exposed SEPARATELY as
# the advisory `second_person_address` rather than folded into direct_mention.

MENTION_RE = re.compile(r"@([A-Za-z]+_\d+)")

# second-person cues; deliberately advisory only
SECOND_PERSON_RE = re.compile(
    r"\b(?:can|could|will|would|are|did|do)\s+you\b"
    r"|\byou\s+(?:need|must|should|have\s+to|are\s+required)\b"
    r"|\b(?:pls|please|kindly)\s+(?:confirm|reply|send|share|revert|respond|check)\b"
    r"|\breply\s+(?:to\s+)?(?:me|this|asap)\b",
    re.IGNORECASE,
)


def find_mentions(text: Optional[str]) -> List[str]:
    """All '@<id>' handles in the text, in order of appearance."""
    return MENTION_RE.findall(text or "")


def detect_direct_mention(text: Optional[str], user_id: Optional[str]) -> bool:
    """True when the text explicitly '@'-mentions this recipient's user_id."""
    if not text or not user_id:
        return False
    return user_id in find_mentions(text)


def detect_second_person(text: Optional[str]) -> bool:
    """Advisory: does the text address a reader in the second person?"""
    return bool(SECOND_PERSON_RE.search(text or ""))


def domains_mismatch(official: Optional[str],
                     used: Optional[str]) -> bool:
    """True when the sending domain differs from the brand's official domain.

    Verified across all 110 business accounts: 82 exact matches, 23 clear
    lookalikes (phonepe.com vs phonepe-rewards.in), ZERO legitimate subdomains
    of the official domain, 5 rows where one side is blank. So plain
    case-insensitive inequality is correct here and no subdomain carve-out is
    needed. Not comparable (either side blank) -> False, with the ambiguity
    surfaced separately as `domain_comparable`.
    """
    if not official or not used:
        return False
    return official.strip().lower() != used.strip().lower()


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def _compact(obj: Any) -> Any:
    """Recursively drop None / empty containers so prompts stay readable."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cv = _compact(v)
            if cv is None or cv == {} or cv == [] or cv == "":
                continue
            out[k] = cv
        return out
    if isinstance(obj, list):
        return [_compact(v) for v in obj if v is not None]
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M")
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


@dataclass
class MessageContext:
    """Everything known about one incoming message, assembled once."""

    # -- message facts ------------------------------------------------------
    message_id: str
    user_id: Optional[str]
    conversation_type: Optional[str]
    created_at: Optional[datetime]
    message_text: Optional[str]          # verbatim, from .raw
    media_type: Optional[str]
    media_id: Optional[str]
    media_file: Optional[str]            # resolved path, or None
    forwarded_count: Optional[int]

    # -- sender -------------------------------------------------------------
    sender_kind: str                     # user | group | business | unknown
    sender_user_id: Optional[str] = None
    sender_profile: Optional[Dict[str, Any]] = None   # users.csv row, if a user

    # -- business sender ----------------------------------------------------
    business_id: Optional[str] = None
    business: Optional[Dict[str, Any]] = None
    domain_mismatch: bool = False
    domain_comparable: bool = True

    # -- user <-> business relationship -------------------------------------
    business_relationship: Optional[Dict[str, Any]] = None
    opted_out: bool = False
    has_business_relationship: bool = False

    # -- recipient ----------------------------------------------------------
    recipient: Optional[Dict[str, Any]] = None
    in_dnd: bool = False
    dnd_window: Optional[str] = None
    daily_load: Optional[Dict[str, Any]] = None           # exact date; see below
    daily_load_baseline: Optional[Dict[str, Any]] = None  # 14-day prior window

    # -- group --------------------------------------------------------------
    group_id: Optional[str] = None
    group: Optional[Dict[str, Any]] = None
    membership: Optional[Dict[str, Any]] = None
    group_muted: bool = False
    group_role: Optional[str] = None

    # -- text signals -------------------------------------------------------
    direct_mention: bool = False
    mentions: List[str] = field(default_factory=list)
    second_person_address: bool = False
    is_forwarded: bool = False

    # -- retrieval keys (for Phase 4) ---------------------------------------
    sender_key: Optional[str] = None
    sender_key_all: List[str] = field(default_factory=list)

    # -- slots for later phases (intentionally empty here) ------------------
    media_analysis: Optional[Dict[str, Any]] = None   # Phase 3 fills
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # Phase 4 fills

    # -- rendering ----------------------------------------------------------

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact nested dict for an LLM prompt. Null/empty fields omitted."""
        d: Dict[str, Any] = {
            "message": {
                "message_id": self.message_id,
                "conversation_type": self.conversation_type,
                "created_at": self.created_at,
                "text": self.message_text,
                "media_type": self.media_type,
                "forwarded_count": self.forwarded_count,
                "is_forwarded": self.is_forwarded or None,
            },
            "sender": {
                "kind": self.sender_kind,
                "user_id": self.sender_user_id,
                "user_stats": self.sender_profile,
                "business": self.business,
                "domain_mismatch": self.domain_mismatch or None,
                # only meaningful for a business sender; omitted otherwise
                "domain_not_comparable": (
                    True if (self.business is not None
                             and not self.domain_comparable) else None),
            },
            "business_relationship": (
                dict(self.business_relationship or {}, opted_out=self.opted_out)
                if self.business_relationship else
                ({"known_to_user": False} if self.business_id else None)
            ),
            "recipient": {
                "user_id": self.user_id,
                "stats_30d": self.recipient,
                "do_not_disturb_window": self.dnd_window,
                "in_do_not_disturb_now": self.in_dnd or None,
                "notification_load_today": self.daily_load,
                "notification_load_baseline": self.daily_load_baseline,
            },
            "group": {
                "group_id": self.group_id,
                "profile": self.group,
                "this_user_membership": self.membership,
                "muted_by_user": self.group_muted or None,
                "role": self.group_role,
            },
            "signals": {
                "direct_mention_of_recipient": self.direct_mention or None,
                "mentions": self.mentions,
                "second_person_address": self.second_person_address or None,
            },
            # filled by later phases; omitted while empty
            "media_analysis": self.media_analysis,
            "evidence": self.evidence,
        }
        return _compact(d)

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_prompt_dict(), indent=indent, ensure_ascii=False,
                          default=str)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def _row_dict(rec: Optional[Record], drop: tuple = ()) -> Optional[Dict[str, Any]]:
    """Record -> plain dict with None values and joined-on keys dropped."""
    if rec is None:
        return None
    return {k: v for k, v in rec.as_dict().items()
            if v is not None and k not in drop}


def daily_load_baseline(ds: Dataset, user_id: Optional[str]
                        ) -> Optional[Dict[str, Any]]:
    """The recipient's notification load over the 14 days preceding the batch.

    Verified: daily_notification_summary.csv covers 2026-07-04..07-17 for all
    54 users (14 days each), while messages.csv runs 2026-07-18..07-31 -- the
    two windows do NOT overlap at all. So an exact same-date lookup can never
    hit, by construction; the table is a *baseline profile* of how heavily
    notified and how dismissal-prone each user is, not a same-day counter.
    This aggregates it into that profile. Returns None for an unknown user.
    """
    rows = ds.daily_by_user.get(user_id or "", [])
    if not rows:
        return None
    sent = [r["notifications_sent"] or 0 for r in rows]
    dism = [r["notifications_dismissed"] or 0 for r in rows]
    total_sent = sum(sent)
    return {
        "days_observed": len(rows),
        "window": "%s..%s" % (min(r["date"] for r in rows),
                              max(r["date"] for r in rows)),
        "avg_notifications_per_day": round(total_sent / len(rows), 2),
        "peak_notifications_in_a_day": max(sent),
        "total_notifications": total_sent,
        "total_dismissed": sum(dism),
        "dismissal_rate": (round(sum(dism) / total_sent, 3)
                           if total_sent else None),
    }


def build_context(ds: Dataset, message: Record) -> MessageContext:
    """Assemble the full context bundle for one message.

    Works unchanged on rows from messages.csv and sample_messages.csv (the
    sample file is a superset of the input schema -- the extra answer columns
    are simply not read here). Every foreign-key lookup is None-safe.
    """
    uid = message.get("user_id")
    gid = message.get("group_id")
    bid = message.get("business_id")
    sid = message.get("sender_user_id")
    created = message.get("created_at")
    # verbatim text, exactly as it appears in the CSV
    text = message.raw.get("message_text") or None

    # -- sender kind -------------------------------------------------------
    if bid:
        sender_kind = "business"
    elif gid:
        sender_kind = "group"
    elif sid:
        sender_kind = "user"
    else:
        sender_kind = "unknown"

    # -- business ----------------------------------------------------------
    biz = ds.get_business(bid)
    official = biz["official_domain"] if biz else None
    used = biz["domain_used_by_sender"] if biz else None
    mismatch = domains_mismatch(official, used)
    comparable = bool(biz) and bool(official) and bool(used)

    rel = ds.get_user_business(uid, bid)
    # allows_promotions is 0/1 -> bool; an explicit opt-out timestamp also counts
    opted_out = bool(rel) and (
        rel["allows_promotions"] is False or rel["promotions_opted_out_at"] is not None
    )

    # -- recipient ---------------------------------------------------------
    user = ds.get_user(uid)
    daily = ds.get_daily_summary(uid, created.date() if created else None)

    # -- group -------------------------------------------------------------
    group = ds.get_group(gid)
    member = ds.get_member(gid, uid)

    # -- text signals ------------------------------------------------------
    mentions = find_mentions(text)
    fwd = message.get("forwarded_count") or 0

    return MessageContext(
        message_id=message["message_id"],
        user_id=uid,
        conversation_type=message.get("conversation_type"),
        created_at=created,
        message_text=text,
        media_type=message.get("media_type"),
        media_id=message.get("media_id"),
        media_file=ds.media_path(message.get("media_id"), message.get("media_type")),
        forwarded_count=message.get("forwarded_count"),

        sender_kind=sender_kind,
        sender_user_id=sid,
        sender_profile=_row_dict(ds.get_user(sid), drop=("user_id",)),

        business_id=bid,
        business=_row_dict(biz, drop=("business_id",)),
        domain_mismatch=mismatch,
        domain_comparable=comparable,

        business_relationship=_row_dict(rel, drop=("user_id", "business_id")),
        opted_out=opted_out,
        has_business_relationship=rel is not None,

        recipient=_row_dict(user, drop=("user_id", "do_not_disturb_window")),
        in_dnd=ds.user_in_dnd(uid, created),
        dnd_window=user["do_not_disturb_window"] if user else None,
        daily_load=_row_dict(daily, drop=("user_id", "date")),
        daily_load_baseline=daily_load_baseline(ds, uid),

        group_id=gid,
        group=_row_dict(group, drop=("group_id",)),
        membership=_row_dict(member, drop=("group_id", "user_id")),
        group_muted=bool(member and member["group_muted_by_user"]),
        group_role=member["role"] if member else None,

        direct_mention=detect_direct_mention(text, uid),
        mentions=mentions,
        second_person_address=detect_second_person(text),
        is_forwarded=fwd > 0,

        sender_key=primary_sender_key(message),
        sender_key_all=sender_keys(message),
    )


def build_all(ds: Dataset, messages: Optional[List[Record]] = None
              ) -> List[MessageContext]:
    """Build contexts for every message, source order preserved."""
    rows = ds.messages if messages is None else messages
    return [build_context(ds, m) for m in rows]


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _self_check() -> int:
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
        if detail:
            print("         %s" % detail)
        if not ok:
            failures.append(label)

    print("=" * 74)
    print("PHASE 2 SELF-CHECK - context_builder.py")
    print("=" * 74)
    ds = load_dataset()

    print("\n1. BUILD ALL BUNDLES")
    ctxs = build_all(ds)
    samples = build_all(ds, ds.sample_messages)
    check("built %d contexts from messages.csv" % len(ctxs), len(ctxs) == 110)
    check("built %d contexts from sample_messages.csv (same code path)"
          % len(samples), len(samples) == 30)
    check("every bundle has a message_id",
          all(c.message_id for c in ctxs + samples))
    check("to_prompt_dict()/to_json() work on all 140 bundles",
          all(isinstance(c.to_prompt_dict(), dict) and c.to_json()
              for c in ctxs + samples))

    print("\n2. CONVERSATION TYPES AND CONTEXT ATTACHMENT")
    kinds: Dict[str, int] = {}
    for c in ctxs:
        kinds[c.conversation_type] = kinds.get(c.conversation_type, 0) + 1
    for k in sorted(kinds):
        print("     %-10s %3d" % (k, kinds[k]))
    check("conversation_type counts match Phase 0 (group 63/business 30/personal 17)",
          kinds == {"group": 63, "business": 30, "personal": 17}, str(kinds))

    biz = [c for c in ctxs if c.conversation_type == "business"]
    grp = [c for c in ctxs if c.conversation_type == "group"]
    per = [c for c in ctxs if c.conversation_type == "personal"]

    check("all %d business bundles carry a business profile" % len(biz),
          all(c.business for c in biz))
    check("all business bundles have sender_kind == 'business'",
          all(c.sender_kind == "business" for c in biz))
    check("all %d group bundles carry group profile AND membership" % len(grp),
          all(c.group and c.membership for c in grp))
    check("all group bundles have sender_kind == 'group'",
          all(c.sender_kind == "group" for c in grp))
    check("all %d personal bundles have NEITHER group nor business" % len(per),
          all(c.group is None and c.business is None
              and c.group_id is None and c.business_id is None for c in per))
    check("all personal bundles resolved their sender's user profile",
          all(c.sender_profile is not None for c in per))
    check("personal bundles render without crashing",
          all(isinstance(c.to_prompt_dict(), dict) for c in per))
    check("group bundles also carry the individual sender_user_id",
          all(c.sender_user_id for c in grp))
    check("every bundle has a recipient profile",
          all(c.recipient is not None for c in ctxs))
    check("every bundle resolved a sender_key for retrieval",
          all(c.sender_key for c in ctxs))
    n_media = sum(1 for c in ctxs if c.media_id)
    check("all %d media bundles resolved a real file path" % n_media,
          n_media == 23 and all(c.media_file for c in ctxs if c.media_id))
    check("Phase 3 slot empty", all(c.media_analysis is None for c in ctxs))
    check("Phase 4 slot empty", all(c.evidence == [] for c in ctxs))
    check("empty slots omitted from prompt dict",
          all("media_analysis" not in c.to_prompt_dict()
              and "evidence" not in c.to_prompt_dict() for c in ctxs))

    print("\n3. DERIVED BOOLEANS (actual counts over 110)")
    flags = ["domain_mismatch", "opted_out", "group_muted", "in_dnd",
             "direct_mention", "second_person_address", "is_forwarded"]
    counts = {f: sum(1 for c in ctxs if getattr(c, f)) for f in flags}
    for f in flags:
        print("     %-24s %3d / 110" % (f, counts[f]))
    check("in_dnd == 8/110 (matches Phase 1)", counts["in_dnd"] == 8,
          "got %d" % counts["in_dnd"])
    check("domain_mismatch only ever set on business bundles",
          all(c.conversation_type == "business"
              for c in ctxs if c.domain_mismatch))
    check("group_muted only ever set on group bundles",
          all(c.conversation_type == "group" for c in ctxs if c.group_muted))
    check("direct_mention only ever set on group bundles",
          all(c.conversation_type == "group" for c in ctxs if c.direct_mention))
    check("opted_out only ever set where a business relationship exists",
          all(c.has_business_relationship for c in ctxs if c.opted_out))
    print("     sample_messages.csv flag counts (30 rows): %s"
          % {f: sum(1 for c in samples if getattr(c, f)) for f in flags})

    print("\n4. DIRECT-MENTION HONESTY CHECK")
    print("     users.csv columns available: %s"
          % sorted(ds.users["u_001"].keys()))
    print("     -> no name/handle/alias column exists for users, so name-based")
    print("        mention matching is NOT implementable. Implemented signal is")
    print("        an explicit '@<user_id>' token matching the recipient.")
    mentioned = [c for c in ctxs if c.mentions]
    print("     bundles containing any @handle : %d" % len(mentioned))
    print("     of those, addressed to the recipient (direct_mention): %d"
          % counts["direct_mention"])
    for c in mentioned:
        print("       %-9s recipient=%s mentions=%s direct=%s group_muted=%s"
              % (c.message_id, c.user_id, c.mentions, c.direct_mention,
                 c.group_muted))
    check("every @mention found resolves to a real user_id in users.csv",
          all(m in ds.users for c in ctxs + samples for m in c.mentions))
    check("direct_mention is a strict subset of bundles carrying a mention",
          counts["direct_mention"] <= len(mentioned))
    both = [c for c in ctxs if c.direct_mention and c.group_muted]
    print("     muted-group bundles WITH a direct mention (Phase 6 override "
          "path): %d %s" % (len(both), [c.message_id for c in both]))

    print("\n5. BLANK FOREIGN KEYS (17 personal messages)")
    blank_fk = [c for c in ctxs if c.group_id is None and c.business_id is None]
    check("17 bundles with blank group_id AND business_id built cleanly",
          len(blank_fk) == 17, "got %d" % len(blank_fk))
    check("their group/business lookups are all None (no KeyError)",
          all(c.group is None and c.membership is None and c.business is None
              and c.business_relationship is None for c in blank_fk))
    check("their derived booleans default to False, not None",
          all(c.group_muted is False and c.domain_mismatch is False
              and c.opted_out is False for c in blank_fk))
    nobiz = [c for c in biz if not c.has_business_relationship]
    print("     business bundles with NO user_business_history row: %d %s"
          % (len(nobiz), [c.message_id for c in nobiz][:8]))
    check("those still render (shown as known_to_user: false)",
          all(c.to_prompt_dict().get("business_relationship", {})
              .get("known_to_user") is False for c in nobiz))
    nodaily = [c for c in ctxs if c.daily_load is None]
    dmin = min(r["date"] for r in ds.daily_summary.values())
    dmax = max(r["date"] for r in ds.daily_summary.values())
    mmin = min(c.created_at.date() for c in ctxs)
    mmax = max(c.created_at.date() for c in ctxs)
    print("     daily_notification_summary covers %s..%s" % (dmin, dmax))
    print("     messages.csv covers             %s..%s" % (mmin, mmax))
    check("the two date windows do NOT overlap, so exact-date daily_load is "
          "None for all 110 by construction (not a bug)",
          len(nodaily) == 110 and dmax < mmin,
          "exact-date misses=%d" % len(nodaily))
    check("every bundle still carries a 14-day daily_load_baseline instead",
          all(c.daily_load_baseline for c in ctxs))
    check("baseline observes 14 days per user",
          all(c.daily_load_baseline["days_observed"] == 14 for c in ctxs))
    ex = ctxs[0]
    print("     baseline for %s: %s" % (ex.user_id, ex.daily_load_baseline))

    print("\n6. EXAMPLE BUNDLES")
    for label, pool in (("PERSONAL", per), ("GROUP", grp), ("BUSINESS", biz)):
        # prefer an interesting one: media / mention / mismatch
        pick = next((c for c in pool if c.direct_mention or c.domain_mismatch
                     or c.media_id), pool[0])
        print("\n  --- %s example: %s ---" % (label, pick.message_id))
        print(pick.to_json(indent=2))

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
