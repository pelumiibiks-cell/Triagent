""" Evidence retrieval
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer import Dataset, Record, load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# tuning constants (named, never inline magic numbers)
# ---------------------------------------------------------------------------

# relative contribution of each signal to the final score; these sum to 1.0
W_SENDER_MATCH = 0.45      # same business / group / person -- strongest prior
W_LEXICAL = 0.40           # topical similarity, IDF-weighted overlap
W_RECENCY = 0.15           # tiebreaker, deliberately the weakest

# recency decays exponentially with age; at HALF_LIFE_DAYS the term is 0.5
HALF_LIFE_DAYS = 45.0

# a candidate must clear this to be cited at all. Below it we return nothing
# rather than emit a weak citation.
RELEVANCE_FLOOR = 0.30

# a same-user-only candidate (no sender overlap) must clear this higher bar,
# since "same user, unrelated sender" is a much weaker claim of relevance
WEAK_MATCH_FLOOR = 0.55

# sample rows cite 1 id (25/30), 2 ids (3/30) or none (2/30) -- so 1-2 is
# typical and 3 is the hard cap. A runner-up must be nearly as good as the
# best hit to be worth citing alongside it.
MAX_EVIDENCE = 3
RUNNER_UP_RATIO = 0.85     # second/third id must score >= this * top score

# ...and must share real content, not merely the same sender. Without this a
# same-sender row scores 0.45 on the sender term alone and rides along on an
# entirely unrelated topic.
MIN_RUNNER_UP_LEXICAL = 0.05

# IDF smoothing floor for a token unseen in the history corpus (i.e. maximally
# rare, so maximally informative)
UNSEEN_TOKEN_IDF = 2.0

# NOTE: near-duplicate suppression was implemented here and then deliberately
# REMOVED. It looked obviously right -- message_history holds repeated rows
# (three copies of "Tower B water pressure will be low ..." for one user) and
# citing all three seems wasteful. But measured against the sample gold it cost
# 18 points of hit rate (75% -> 57%), because repetition is precisely what
# several routing decisions rest on: sample_msg_015's gold cites message_0017
# AND message_0018, which are character-identical promos, to justify muting a
# repetitive sender. Citing the duplicates IS the evidence. Do not re-add this.

HISTORY_PREFIX = "message_"

STOPWORDS = frozenset("""
a about above after again all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do
does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same shan she should shouldn so
some such than that the their theirs them themselves then there these they
this those through to too under until up very was wasn we were weren what when
where which while who whom why will with won would wouldn you your yours
yourself yourselves us get got please pls kindly thanks thank hi hello dear
will also may
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------

def tokenize(text: Optional[str]) -> frozenset:
    """Lowercased content tokens, stopwords and 1-char noise removed."""
    if not text:
        return frozenset()
    toks = TOKEN_RE.findall(text.lower())
    return frozenset(t for t in toks if len(t) > 1 and t not in STOPWORDS)


# IDF is computed once over the message_history corpus and memoised.
_IDF: Optional[Dict[str, float]] = None


def build_idf(ds: Dataset, media_cache: Optional[Dict[str, Any]] = None
              ) -> Dict[str, float]:
    """Inverse document frequency over message_history.csv, memoised."""
    global _IDF
    if _IDF is not None:
        return _IDF
    docs = [tokenize(searchable_text(h, media_cache)) for h in ds.history_rows]
    n = len(docs) or 1
    df: Dict[str, int] = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    _IDF = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return _IDF


def _weight(tokens: frozenset, idf: Dict[str, float]) -> float:
    return sum(idf.get(t, UNSEEN_TOKEN_IDF) for t in tokens)


def lexical_similarity(a: frozenset, b: frozenset,
                       idf: Optional[Dict[str, float]] = None) -> float:
    """IDF-weighted overlap coefficient in [0,1].

    Deliberately NOT Jaccard. Jaccard divides by the union, so it is highly
    length-sensitive: attaching Phase 3 media understanding (a full page of
    OCR'd form text, say) to one side inflates the union and crushes the score
    even when every meaningful term matches. Measured on the sample set, that
    bug alone cost 2 retrievals and 6 points of recall.

    The overlap coefficient divides by the *smaller* side instead, so a short
    message still matches a long one on shared content. IDF weighting then
    makes rare, topical terms ("tanker", "consent") count for far more than
    ubiquitous ones ("update", "please").
    """
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    if idf is None:
        return len(inter) / min(len(a), len(b))
    denom = min(_weight(a, idf), _weight(b, idf))
    return _weight(inter, idf) / denom if denom else 0.0


def searchable_text(row: Record, media_cache: Optional[Dict[str, Any]] = None
                    ) -> str:
    """Message text, plus Phase 3 media understanding when the text is blank.

    Voice-note rows have empty message_text by design (Phase 0: the 8 blank
    texts in messages.csv are exactly the voice messages), so without this a
    voice or image message would have no lexical signal at all.
    """
    parts = [row.raw.get("message_text") or ""]
    mid = row["media_id"]
    if mid and media_cache:
        rec = media_cache.get(mid) or {}
        if "error" not in rec:
            for key in ("transcript", "visible_text", "summary",
                        "apparent_intent"):
                val = rec.get(key)
                if val:
                    parts.append(str(val))
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# evidence item
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """One retrieved historical message plus the user's reaction to it."""

    message_id: str
    user_id: str
    created_at: Optional[datetime]
    text: str
    conversation_type: Optional[str]
    media_type: Optional[str] = None

    # scoring breakdown, kept inspectable
    score: float = 0.0
    sender_match: bool = False
    matched_sender_key: Optional[str] = None
    lexical: float = 0.0
    recency: float = 0.0
    age_days: Optional[float] = None

    # joined from message_events -- the strongest personalisation signal
    reaction: Dict[str, Any] = field(default_factory=dict)

    def reaction_summary(self) -> str:
        """Compact human/LLM-readable reaction string."""
        if not self.reaction:
            return "no recorded reaction"
        order = [("reported", "reported"), ("muted_after", "muted after"),
                 ("replied", "replied"), ("opened", "opened"),
                 ("dismissed", "dismissed")]
        hit = [label for key, label in order if self.reaction.get(key)]
        if not hit:
            return "ignored (not opened)"
        mins = self.reaction.get("reaction_time_minutes")
        s = ", ".join(hit)
        if mins is not None and self.reaction.get("opened"):
            s += " (after %s min)" % mins
        return s

    def to_prompt_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "message_id": self.message_id,
            "date": (self.created_at.strftime("%Y-%m-%d %H:%M")
                     if self.created_at else None),
            "text": (self.text or "").strip()[:400] or None,
            "user_reaction": self.reaction_summary(),
            "same_sender": self.sender_match or None,
            "relevance": round(self.score, 3),
        }
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def _reaction_for(ds: Dataset, message_id: str) -> Dict[str, Any]:
    ev = ds.get_event(message_id)
    if ev is None:
        return {}
    return {
        "opened": ev["message_opened"],
        "replied": ev["message_replied"],
        "dismissed": ev["notification_dismissed"],
        "muted_after": ev["muted_after_message"],
        "reported": ev["message_reported"],
        "reaction_time_minutes": ev["reaction_time_minutes"],
    }


def _recency_score(when: Optional[datetime],
                   now: Optional[datetime]) -> tuple:
    """Exponential decay on age. Returns (score, age_days)."""
    if when is None or now is None:
        return 0.0, None
    age_days = max(0.0, (now - when).total_seconds() / 86400.0)
    return math.exp(-age_days / HALF_LIFE_DAYS * math.log(2)), age_days


def retrieve_evidence(ds: Dataset,
                      ctx: Any,
                      k: int = MAX_EVIDENCE,
                      media_cache: Optional[Dict[str, Any]] = None
                      ) -> List[EvidenceItem]:
    """Rank this user's history for relevance to `ctx`, then filter.

    `ctx` is a Phase 2 MessageContext (anything exposing user_id, created_at,
    message_id, sender_key_all, media_id). Returns 0..k items, newest-and-most
    relevant first. An empty list is a valid, meaningful result.
    """
    uid = getattr(ctx, "user_id", None)
    if not uid:
        return []

    pool = ds.get_history_for_user(uid)
    if not pool:
        return []

    now = getattr(ctx, "created_at", None)
    own_keys = set(getattr(ctx, "sender_key_all", []) or [])
    routed_id = getattr(ctx, "message_id", None)

    # query text: the message itself plus its media understanding
    query_text = getattr(ctx, "message_text", "") or ""
    analysis = getattr(ctx, "media_analysis", None)
    if analysis and "error" not in analysis:
        for key in ("transcript", "visible_text", "summary", "apparent_intent"):
            if analysis.get(key):
                query_text += "\n" + str(analysis[key])
    q_tokens = tokenize(query_text)
    idf = build_idf(ds, media_cache)

    scored: List[EvidenceItem] = []
    for h in pool:
        hid = h["message_id"]
        if hid == routed_id:            # cannot cite itself
            continue

        h_keys = set(_sender_keys_of(h))
        shared = own_keys & h_keys
        sender_match = bool(shared)

        lex = lexical_similarity(
            q_tokens, tokenize(searchable_text(h, media_cache)), idf)
        rec, age = _recency_score(h["created_at"], now)

        score = (W_SENDER_MATCH * (1.0 if sender_match else 0.0)
                 + W_LEXICAL * lex
                 + W_RECENCY * rec)

        scored.append(EvidenceItem(
            message_id=hid,
            user_id=h["user_id"],
            created_at=h["created_at"],
            text=searchable_text(h, media_cache),
            conversation_type=h["conversation_type"],
            media_type=h["media_type"],
            score=score,
            sender_match=sender_match,
            matched_sender_key=sorted(shared)[0] if shared else None,
            lexical=lex,
            recency=rec,
            age_days=round(age, 1) if age is not None else None,
            reaction=_reaction_for(ds, hid),
        ))

    # deterministic ordering: score desc, then recency desc, then id
    scored.sort(key=lambda e: (-e.score,
                               -(e.created_at.timestamp() if e.created_at else 0),
                               e.message_id))

    # relevance floor -- a same-user-only hit must clear a higher bar
    kept = [e for e in scored
            if e.score >= (RELEVANCE_FLOOR if e.sender_match
                           else WEAK_MATCH_FLOOR)]
    if not kept:
        return []

    # Only cite runners-up that are nearly as strong as the best hit AND share
    # real content with it. Sharing only a sender is not evidence of relevance.
    # Duplicates are intentionally NOT suppressed -- see the note above.
    top = kept[0].score
    out = [kept[0]]
    for e in kept[1:k]:
        if (e.score >= RUNNER_UP_RATIO * top
                and e.lexical >= MIN_RUNNER_UP_LEXICAL):
            out.append(e)
    return out[:k]


def _sender_keys_of(row: Record) -> List[str]:
    keys = []
    if row["group_id"]:
        keys.append("group:%s" % row["group_id"])
    if row["business_id"]:
        keys.append("business:%s" % row["business_id"])
    if row["sender_user_id"]:
        keys.append("user:%s" % row["sender_user_id"])
    return keys


# ---------------------------------------------------------------------------
# contract enforcement
# ---------------------------------------------------------------------------

class EvidenceContractError(AssertionError):
    """Raised when retrieval would violate the evidence_message_ids contract."""


def assert_contract(ds: Dataset, ctx: Any, items: Sequence[EvidenceItem]) -> None:
    """Mechanically enforce the evidence contract. Raises on any violation."""
    for e in items:
        if not e.message_id.startswith(HISTORY_PREFIX):
            raise EvidenceContractError(
                "evidence id %r for %s is not from message_history.csv "
                "(expected the %r namespace)"
                % (e.message_id, getattr(ctx, "message_id", "?"), HISTORY_PREFIX))
        row = ds.get_history(e.message_id)
        if row is None:
            raise EvidenceContractError(
                "evidence id %r for %s does not exist in message_history.csv "
                "- fabricated id" % (e.message_id, getattr(ctx, "message_id", "?")))
        if row["user_id"] != getattr(ctx, "user_id", None):
            raise EvidenceContractError(
                "evidence id %r belongs to user %s but message %s is for user %s"
                % (e.message_id, row["user_id"], getattr(ctx, "message_id", "?"),
                   getattr(ctx, "user_id", None)))
    ids = [e.message_id for e in items]
    if len(ids) != len(set(ids)):
        raise EvidenceContractError("duplicate evidence ids: %s" % ids)
    if len(ids) > MAX_EVIDENCE:
        raise EvidenceContractError("too many evidence ids (%d > %d): %s"
                                    % (len(ids), MAX_EVIDENCE, ids))


def render_evidence_ids(items: Sequence[EvidenceItem]) -> str:
    """The output.csv cell value: semicolon-separated ids, or literal 'none'."""
    if not items:
        return "none"
    return ";".join(e.message_id for e in items)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def attach_evidence(ds: Dataset, contexts: Sequence[Any],
                    media_cache: Optional[Dict[str, Any]] = None,
                    k: int = MAX_EVIDENCE) -> int:
    """Fill the Phase 2 `evidence` slot on each context. Returns items attached."""
    total = 0
    for ctx in contexts:
        items = retrieve_evidence(ds, ctx, k=k, media_cache=media_cache)
        assert_contract(ds, ctx, items)
        ctx.evidence = [e.to_prompt_dict() for e in items]
        ctx.evidence_items = items          # full objects for Phase 5/6
        total += len(items)
    return total


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _gold_ids(row: Record) -> List[str]:
    raw = row["evidence_message_ids"] or ""
    return [x.strip() for x in raw.split(";")
            if x.strip() and x.strip() != "none"]


def _self_check() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
        if detail:
            print("         %s" % detail)
        if not ok:
            failures.append(label)

    from context_builder import build_all
    from media_analysis import attach_to_contexts, load_cache

    print("=" * 74)
    print("PHASE 4 SELF-CHECK - evidence.py")
    print("=" * 74)
    ds = load_dataset()
    cache = load_cache()

    ctxs = build_all(ds)
    samples = build_all(ds, ds.sample_messages)
    attach_to_contexts(ctxs, cache)
    attach_to_contexts(samples, cache)

    print("\nscoring weights: sender=%.2f lexical=%.2f recency=%.2f"
          % (W_SENDER_MATCH, W_LEXICAL, W_RECENCY))
    print("floors: same-sender>=%.2f  same-user-only>=%.2f  runner-up>=%.0f%% of top"
          % (RELEVANCE_FLOOR, WEAK_MATCH_FLOOR, RUNNER_UP_RATIO * 100))
    print("half-life: %.0f days   max evidence: %d" % (HALF_LIFE_DAYS, MAX_EVIDENCE))

    # -- 1. contract --------------------------------------------------------
    print("\n1. CONTRACT ENFORCEMENT (110 real + 30 sample)")
    batch_ids = {m["message_id"] for m in ds.messages}
    sample_ids = {m["message_id"] for m in ds.sample_messages}
    all_items: Dict[str, List[EvidenceItem]] = {}
    violations = 0
    for pool, label in ((ctxs, "messages"), (samples, "sample")):
        for ctx in pool:
            items = retrieve_evidence(ds, ctx, media_cache=cache)
            try:
                assert_contract(ds, ctx, items)
            except EvidenceContractError as exc:
                violations += 1
                print("    VIOLATION: %s" % exc)
            all_items[ctx.message_id] = items
    check("assert_contract passed for all 140 messages", violations == 0)

    every = [e for items in all_items.values() for e in items]
    check("every returned id is in the message_#### namespace (%d ids)"
          % len(every),
          all(e.message_id.startswith(HISTORY_PREFIX) for e in every))
    bad_ns = [e.message_id for e in every
              if e.message_id in batch_ids or e.message_id in sample_ids]
    check("zero ids leaked from messages.csv or sample_messages.csv",
          not bad_ns, str(bad_ns[:5]))
    check("every returned id exists in message_history.csv (nothing fabricated)",
          all(ds.get_history(e.message_id) is not None for e in every))
    mismatched = [(mid, e.message_id) for mid, items in all_items.items()
                  for e in items
                  if ds.get_history(e.message_id)["user_id"] !=
                  next(c for c in ctxs + samples if c.message_id == mid).user_id]
    check("zero returned ids whose history user_id differs from the routed "
          "message's user_id", not mismatched, str(mismatched[:5]))
    check("every item carries a joined message_events reaction",
          all(e.reaction for e in every))
    check("no item exceeds the %d-id cap" % MAX_EVIDENCE,
          all(len(v) <= MAX_EVIDENCE for v in all_items.values()))

    # -- 2. distribution ----------------------------------------------------
    print("\n2. EVIDENCE COUNT DISTRIBUTION")
    for pool, label, n in ((ctxs, "messages.csv", 110),
                           (samples, "sample_messages.csv", 30)):
        dist: Dict[int, int] = {}
        for ctx in pool:
            c = len(all_items[ctx.message_id])
            dist[c] = dist.get(c, 0) + 1
        line = "  ".join("%d ids: %d" % (k, dist.get(k, 0))
                         for k in range(0, MAX_EVIDENCE + 1))
        print("  %-22s %s   (n=%d)" % (label, line, n))
    gold_dist: Dict[int, int] = {}
    for s in ds.sample_messages:
        c = len(_gold_ids(s))
        gold_dist[c] = gold_dist.get(c, 0) + 1
    print("  %-22s %s   (n=30)" % ("sample GOLD",
                                   "  ".join("%d ids: %d" % (k, gold_dist.get(k, 0))
                                             for k in range(0, MAX_EVIDENCE + 1))))

    # -- 3. dry-run against sample gold -------------------------------------
    print("\n3. RETRIEVAL QUALITY vs sample_messages.csv GOLD (the only ground truth)")
    exact = overlap_only = miss = 0
    gold_none_agree = gold_none_total = 0
    recall_num = recall_den = 0
    rows = []
    for s, ctx in zip(ds.sample_messages, samples):
        gold = set(_gold_ids(s))
        got = {e.message_id for e in all_items[ctx.message_id]}
        if not gold:
            gold_none_total += 1
            if not got:
                gold_none_agree += 1
            rows.append((ctx.message_id, "none", sorted(got) or ["none"],
                         "AGREE" if not got else "over-cited"))
            continue
        recall_den += len(gold)
        recall_num += len(gold & got)
        if got == gold:
            exact += 1
            verdict = "EXACT"
        elif gold & got:
            overlap_only += 1
            verdict = "partial"
        else:
            miss += 1
            verdict = "MISS"
        rows.append((ctx.message_id, sorted(gold), sorted(got) or ["none"], verdict))

    scored_n = 30 - gold_none_total
    print("  rows with gold evidence      : %d" % scored_n)
    print("  exact set match              : %d/%d  (%.0f%%)"
          % (exact, scored_n, 100.0 * exact / scored_n))
    print("  partial overlap (non-empty)  : %d/%d  (%.0f%%)"
          % (overlap_only, scored_n, 100.0 * overlap_only / scored_n))
    print("  total miss (no overlap)      : %d/%d  (%.0f%%)"
          % (miss, scored_n, 100.0 * miss / scored_n))
    print("  hit rate (exact + partial)   : %d/%d  (%.0f%%)"
          % (exact + overlap_only, scored_n,
             100.0 * (exact + overlap_only) / scored_n))
    print("  gold-id recall               : %d/%d ids (%.0f%%)"
          % (recall_num, recall_den, 100.0 * recall_num / recall_den))
    print("  gold == 'none' agreement     : %d/%d"
          % (gold_none_agree, gold_none_total))

    print("\n  per-row detail:")
    print("    %-16s %-30s %-30s %s" % ("message_id", "gold", "retrieved", "verdict"))
    for mid, gold, got, verdict in rows:
        g = ";".join(gold) if isinstance(gold, list) else gold
        print("    %-16s %-30s %-30s %s" % (mid, g, ";".join(got), verdict))

    # -- 4. worked examples -------------------------------------------------
    print("\n4. WORKED EXAMPLES")
    picks = []
    for mid, gold, got, verdict in rows:
        if verdict in ("EXACT", "partial", "MISS", "AGREE") and len(picks) < 4:
            if verdict not in [p[3] for p in picks]:
                picks.append((mid, gold, got, verdict))
    # add one real (non-sample) message
    real = next(c for c in ctxs if all_items[c.message_id])
    for mid, gold, got, verdict in picks:
        ctx = next(c for c in samples if c.message_id == mid)
        src = next(s for s in ds.sample_messages if s["message_id"] == mid)
        print("\n  --- %s  [%s]  gold=%s" % (mid, verdict,
                                             ";".join(gold) if isinstance(gold, list) else gold))
        print("      conversation_type=%s  sender_key=%s"
              % (ctx.conversation_type, ctx.sender_key))
        print("      text: %r" % ((ctx.message_text or "")[:170]))
        if ctx.media_analysis:
            print("      media: %r"
                  % (ctx.media_analysis.get("transcript")
                     or ctx.media_analysis.get("summary"))[:150])
        print("      gold action/type: %s / %s" % (src["action"], src["message_type"]))
        if not all_items[mid]:
            print("      -> retrieved NOTHING (all candidates below floor)")
        for e in all_items[mid]:
            print("      -> %s  score=%.3f (sender=%s lex=%.3f rec=%.3f age=%sd)"
                  % (e.message_id, e.score, e.sender_match, e.lexical,
                     e.recency, e.age_days))
            print("         text: %r" % (e.text or "")[:150].replace("\n", " "))
            print("         reaction: %s" % e.reaction_summary())

    print("\n  --- %s  [real batch message] ---" % real.message_id)
    print("      conversation_type=%s  sender_key=%s"
          % (real.conversation_type, real.sender_key))
    print("      text: %r" % ((real.message_text or "")[:170]))
    for e in all_items[real.message_id]:
        print("      -> %s  score=%.3f (sender=%s lex=%.3f rec=%.3f)"
              % (e.message_id, e.score, e.sender_match, e.lexical, e.recency))
        print("         text: %r" % (e.text or "")[:150].replace("\n", " "))
        print("         reaction: %s" % e.reaction_summary())

    # -- 5. wiring ----------------------------------------------------------
    print("\n5. WIRING INTO PHASE 2 BUNDLES")
    n = attach_evidence(ds, ctxs, cache)
    print("     attached %d evidence items across 110 bundles" % n)
    withev = [c for c in ctxs if c.evidence]
    check("evidence slot populated on %d/110 bundles" % len(withev), bool(withev))
    check("to_prompt_dict() surfaces evidence with text/date/reaction",
          all({"message_id", "user_reaction"} <= set(c.evidence[0])
              for c in withev))
    ex = withev[0]
    print("     example rendering for %s:" % ex.message_id)
    for e in ex.to_prompt_dict()["evidence"]:
        print("       %s" % e)
    print("     render_evidence_ids -> %r"
          % render_evidence_ids(all_items[ex.message_id]))
    empty = next((c for c in ctxs if not c.evidence), None)
    if empty:
        print("     example empty rendering (%s) -> %r"
              % (empty.message_id, render_evidence_ids([])))

    print("\n" + "=" * 74)
    if failures:
        print("SELF-CHECK FAILED - %d failing check(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SELF-CHECK PASSED - contract checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
