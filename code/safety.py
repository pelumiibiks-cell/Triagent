"""Phase 6 - Safety overrides.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer import load_dataset  # noqa: E402
from context_builder import build_all  # noqa: E402
from media_analysis import attach_to_contexts, load_cache  # noqa: E402


# thresholds


# A brand-new account with many reports, sending from a domain that is not the
# brand's official one, is impersonation regardless of message content.
IMPERSONATION_MIN_REPORTS = 20
IMPERSONATION_MAX_AGE_DAYS = 90

# Types that may be promoted out of a muted group. Chain forwards, greetings,
# promotions and junk stay muted even when they contain an @mention -- a
# blessing-chain that tags you is not a genuine direct mention.
NOTIFY_ELIGIBLE_TYPES = ("personal", "urgent", "event", "payment",
                         "business_update")

# ---------------------------------------------------------------------------
# fraud patterns
# ---------------------------------------------------------------------------
#
# Deliberately high-precision. Each requires a fraud ACT (share/confirm/send a
# secret, redirect a payment), not merely a topic word -- "OTP" alone appears
# in legitimate bank notices, and a bank's own anti-fraud poster is full of the
# word "scam". Phase 3 already proved that trap is real (img_026).

CREDENTIAL_REQUEST = re.compile(
    r"(?:share|send|reply\s+with|confirm|enter|provide|give|tell\s+me|forward)"
    r"[^.!?\n]{0,60}"
    r"(?:otp|one[\s-]?time\s?password|verification\s+code|login\s+code|"
    r"security\s+code|\b\d\s?digit\s+code\b|password|\bpin\b|cvv)"
    r"|(?:otp|verification\s+code|login\s+code)[^.!?\n]{0,40}"
    r"(?:share|send|reply|confirm|enter|provide)",
    re.IGNORECASE)

PAYMENT_REDIRECTION = re.compile(
    r"(?:pay|transfer|deposit|send\s+money|clearance\s+amount|fee|"
    r"processing\s+charge|reattempt)"
    r"[^.!?\n]{0,80}"
    r"(?:link|url|http|www\.|scan|qr|wallet|upi|account\s+number|bank\s+details)"
    r"|(?:scan)[^.!?\n]{0,40}(?:qr)[^.!?\n]{0,40}(?:pay|amount)"
    r"|(?:bank\s+details|account\s+details)[^.!?\n]{0,60}"
    r"(?:fill|send|share|submit|screenshot)",
    re.IGNORECASE)

PRIZE_BAIT = re.compile(
    r"(?:congrat\w*|you\s+have\s+won|winner|selected\s+for|lucky\s+draw|"
    r"reward|prize|voucher|cashback\s+bonus|lottery)"
    r"[^.!?\n]{0,90}"
    r"(?:claim|expire|hurry|today|immediately|before|last\s+chance|urgent)",
    re.IGNORECASE)

ACCOUNT_THREAT = re.compile(
    r"(?:account|card|access|kyc|sim|number)\s+(?:will\s+be\s+)?"
    r"(?:blocked|suspended|deactivat\w*|expire\w*|clos\w*|frozen|locked)"
    r"[^.!?\n]{0,90}"
    r"(?:now|today|immediately|within|urgent|verify|login|confirm|update|"
    r"click|link|reply)",
    re.IGNORECASE)

# An instruction aimed at the routing system itself, embedded in message text.
PROMPT_INJECTION = re.compile(
    r"(?:routing\s+override|set\s+action\s*=|ignore\s+(?:all\s+)?previous|"
    r"disregard\s+(?:the\s+)?(?:above|instructions)|system\s+prompt|"
    r"you\s+are\s+now|confidence\s*=\s*1)",
    re.IGNORECASE)

CHAIN_FORWARD = re.compile(
    r"(?:forward|share|send)\s+(?:this\s+)?(?:to\s+)?"
    r"(?:\d+\s+(?:people|friends|groups|contacts)|all\s+(?:your\s+)?"
    r"(?:groups|contacts|friends)|everyone)"
    r"|do\s+not\s+break\s+the\s+chain|luck\s+changes\s+when\s+you\s+share",
    re.IGNORECASE)


# Anti-fraud WARNINGS use the same vocabulary as fraud itself. A society admin
# writing "don't use any payment link shared by residents" is protecting the
# user, not attacking them -- but a naive "payment ... link" match flags it as
# redirection. This is the text-side twin of the img_026 trap, where a bank's
# own anti-scam poster is full of the word "scammers".
#
# So a sentence whose fraud match sits in a warning or negated context is not
# counted as a signal.
WARNING_CONTEXT = re.compile(
    r"\b(?:do\s?n[o']?t|don't|never|avoid|beware|ignore|do\s+not|"
    r"no\s+one\s+will|we\s+will\s+never|will\s+never\s+ask|"
    r"not\s+ask\s+for|refrain\s+from|should\s+not|must\s+not|"
    r"be\s+(?:careful|cautious|aware)|fraud\s+awareness|stay\s+safe|"
    r"report\s+(?:it|such|any)|is\s+a\s+scam|are\s+scam)",
    re.IGNORECASE)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def _matches_outside_warnings(pattern: "re.Pattern[str]", text: str) -> bool:
    """True when `pattern` fires in at least one non-warning sentence.

    Evaluated sentence by sentence so that a legitimate message which both
    discusses a payment AND warns against payment links is not flagged.
    """
    for sentence in SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        if pattern.search(sentence) and not WARNING_CONTEXT.search(sentence):
            return True
    return False


def _message_surface(ctx: Any) -> str:
    """All text a fraud check should see: the message plus its media content."""
    parts = [ctx.message_text or ""]
    a = ctx.media_analysis or {}
    if a and "error" not in a:
        for key in ("transcript", "visible_text", "apparent_intent", "summary"):
            if a.get(key):
                parts.append(str(a[key]))
    return "\n".join(parts)


def scam_signals(ctx: Any) -> List[str]:
    """Named fraud signals present. Empty list means no hard risk detected."""
    text = _message_surface(ctx)
    found: List[str] = []

    if _matches_outside_warnings(CREDENTIAL_REQUEST, text):
        found.append("credential_or_otp_request")
    if _matches_outside_warnings(PAYMENT_REDIRECTION, text):
        found.append("payment_redirection")
    if _matches_outside_warnings(PRIZE_BAIT, text):
        found.append("prize_or_reward_bait")
    if _matches_outside_warnings(ACCOUNT_THREAT, text):
        found.append("account_threat_with_urgency")
    # An injection attempt is never legitimate, warning context or not.
    if PROMPT_INJECTION.search(text):
        found.append("prompt_injection_attempt")

    biz = ctx.business
    if biz and ctx.domain_mismatch and not biz.get("verified"):
        reports = biz.get("user_reports_30d") or 0
        age = biz.get("account_age_days")
        if (reports >= IMPERSONATION_MIN_REPORTS
                and age is not None and age <= IMPERSONATION_MAX_AGE_DAYS):
            found.append("brand_impersonation_domain_mismatch")

    # Phase 3 vision flags, but only the unambiguous ones. `has_payment_request`
    # alone is not enough: a charity walkathon's entry fee trips it (img_001).
    a = ctx.media_analysis or {}
    flags = [str(f).lower() for f in (a.get("red_flags") or [])]
    if any("impersonat" in f for f in flags):
        found.append("media_brand_impersonation")
    if a.get("has_qr_code") and a.get("has_payment_request"):
        found.append("media_qr_payment_request")

    return found


def spam_signals(ctx: Any) -> List[str]:
    """Unwanted-bulk signals. Lower severity than scam: mute as `spam`."""
    text = _message_surface(ctx)
    found: List[str] = []
    if CHAIN_FORWARD.search(text):
        found.append("chain_forward_instruction")
    return found


# ---------------------------------------------------------------------------
# override application
# ---------------------------------------------------------------------------

def apply_overrides(ctx: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Phase 6 safety rules to a classified result.

    Returns a NEW dict; never mutates the input. Records every rule that fired
    under `_overrides` so the decision stays auditable.
    """
    out = dict(result)
    applied: List[Dict[str, str]] = []

    scam = scam_signals(ctx)
    spam = spam_signals(ctx)

    # -- rule 1: hard scam override, beats the model and any engagement -----
    if scam:
        if out["action"] != "mute" or out["message_type"] != "scam":
            applied.append({
                "rule": "force_mute_scam",
                "signals": ",".join(scam),
                "was": "%s/%s" % (out["action"], out["message_type"]),
            })
            out["action"] = "mute"
            out["message_type"] = "scam"
            out["reason"] = (
                "Clear fraud signals mean this is muted as a scam regardless "
                "of the user's usual engagement with this sender.")
            out["confidence"] = max(float(out.get("confidence") or 0), 0.88)

    # -- rule 2: chain-forward spam ----------------------------------------
    elif spam and out["action"] != "mute":
        applied.append({
            "rule": "force_mute_spam",
            "signals": ",".join(spam),
            "was": "%s/%s" % (out["action"], out["message_type"]),
        })
        out["action"] = "mute"
        if out["message_type"] not in ("spam", "forward"):
            out["message_type"] = "spam"
        out["reason"] = ("The message is a chain forward asking to be spread "
                         "onward, which this user does not need.")

    # -- rule 3: a muted group must still let a genuine direct mention through
    elif (ctx.group_muted and ctx.direct_mention
            and out["action"] == "mute"
            and out["message_type"] in NOTIFY_ELIGIBLE_TYPES):
        applied.append({
            "rule": "allow_notify_direct_mention_in_muted_group",
            "signals": "direct_mention",
            "was": "%s/%s" % (out["action"], out["message_type"]),
        })
        out["action"] = "notify"
        out["reason"] = ("The user is directly mentioned with content needing "
                         "their attention, so the group mute is overridden.")

    # -- rule 4: genuinely urgent content in a muted group ------------------
    elif (ctx.group_muted and out["message_type"] == "urgent"
            and out["action"] == "mute"):
        applied.append({
            "rule": "allow_notify_urgent_in_muted_group",
            "signals": "urgent_in_muted_group",
            "was": "%s/%s" % (out["action"], out["message_type"]),
        })
        out["action"] = "notify"
        out["reason"] = ("Genuinely urgent content aimed at this user "
                         "overrides the group mute.")

    if applied:
        out["_overrides"] = applied
    return out


def apply_all(contexts: Sequence[Any],
              results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply overrides across a whole batch, keyed by message_id."""
    by_id = {c.message_id: c for c in contexts}
    out: Dict[str, Dict[str, Any]] = {}
    for mid, res in results.items():
        ctx = by_id.get(mid)
        out[mid] = apply_overrides(ctx, res) if ctx is not None else dict(res)
    return out


# ---------------------------------------------------------------------------
# audit / self-check
# ---------------------------------------------------------------------------

def _load_results(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("message_id"):
                out[rec["message_id"]] = rec
    return out


def _main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    import classifier

    sample = "--dry-run" in sys.argv
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
        if detail:
            print("         %s" % detail)
        if not ok:
            failures.append(label)

    print("=" * 74)
    print("PHASE 6 - SAFETY OVERRIDES%s" % ("  [DRY RUN]" if sample else ""))
    print("=" * 74)

    ds = load_dataset()
    cache = load_cache()
    rows = ds.sample_messages if sample else ds.messages
    ctxs = build_all(ds, rows)
    attach_to_contexts(ctxs, cache)

    path = (classifier.DRYRUN_CHECKPOINT if sample
            else classifier.CHECKPOINT_PATH)
    before = _load_results(path)
    if not before:
        print("no classifications found at %s - run classifier.py first" % path)
        return 1
    after = apply_all(ctxs, before)

    changed = [m for m in after if after[m].get("_overrides")]
    print("\n1. OVERRIDE ACTIVITY")
    print("  classified rows      : %d" % len(after))
    print("  rows overridden      : %d" % len(changed))
    rules: Dict[str, int] = {}
    for m in changed:
        for o in after[m]["_overrides"]:
            rules[o["rule"]] = rules.get(o["rule"], 0) + 1
    for r, n in sorted(rules.items()):
        print("     %-46s %d" % (r, n))

    print("\n  detail:")
    for m in sorted(changed):
        o = after[m]["_overrides"][0]
        print("    %-9s %-42s %s -> %s/%s"
              % (m, o["rule"], o["was"], after[m]["action"],
                 after[m]["message_type"]))
        print("        signals: %s" % o["signals"])

    print("\n2. SIGNAL COVERAGE")
    sig_rows = [(c.message_id, scam_signals(c)) for c in ctxs]
    firing = [(m, s) for m, s in sig_rows if s]
    print("  rows with >=1 scam signal: %d/%d" % (len(firing), len(ctxs)))
    counts: Dict[str, int] = {}
    for _, s in firing:
        for x in s:
            counts[x] = counts.get(x, 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("     %-40s %d" % (k, v))

    print("\n3. INVARIANTS")
    check("every scam-signalled row ends as mute/scam",
          all(after[m]["action"] == "mute" and after[m]["message_type"] == "scam"
              for m, s in firing),
          str([(m, after[m]["action"], after[m]["message_type"])
               for m, s in firing
               if after[m]["action"] != "mute"][:5]))
    check("overrides never produce a value outside the enums",
          all(after[m]["action"] in classifier.ACTIONS
              and after[m]["message_type"] in classifier.MESSAGE_TYPES
              for m in after))
    check("overrides never blank a reason",
          all((after[m].get("reason") or "").strip() for m in after))
    check("confidence stays in [0,1]",
          all(0.0 <= float(after[m]["confidence"]) <= 1.0 for m in after))
    check("evidence ids untouched by overrides",
          all(after[m]["evidence_message_ids"] == before[m]["evidence_message_ids"]
              for m in after))
    check("apply_overrides does not mutate its input",
          all("_overrides" not in before[m] for m in before))

    # the anti-scam poster must NOT be caught
    poster = [c for c in ctxs
              if (c.media_analysis or {}).get("media_id") == "img_026"]
    if poster:
        check("HDFC anti-scam awareness poster (img_026) is not flagged",
              all(not scam_signals(c) for c in poster),
              str([(c.message_id, scam_signals(c)) for c in poster]))
    walk = [c for c in ctxs
            if (c.media_analysis or {}).get("media_id") == "img_001"]
    if walk:
        check("charity walkathon entry fee (img_001) is not flagged as scam",
              all(not scam_signals(c) for c in walk),
              str([(c.message_id, scam_signals(c)) for c in walk]))

    # Regression guards: anti-fraud WARNINGS must never be read as fraud.
    # msg_021 is a real society-admin payment reminder that explicitly warns
    # against payment links; a naive "payment ... link" match flagged it and
    # overrode a correct notify. Keep these cases pinned.
    print("\n3b. WARNING-CONTEXT REGRESSION GUARDS")
    guards = [
        ("legit admin reminder warning against payment links",
         "Payment due today. Complete before 5 PM. If already paid, ignore; "
         "receipts will be matched in evening. Please don't use any payment "
         "link shared by residents.", False),
        ("bank fraud-awareness notice",
         "We will never ask for your OTP or password. Report any such request.",
         False),
        ("real OTP harvesting attempt",
         "Your account will be blocked today. Share the OTP you received so "
         "we can complete verification immediately.", True),
        ("real payment redirection",
         "Delivery failed. Pay small reattempt fee at amazonpay-delivery.in "
         "and enter OTP to release package.", True),
    ]

    class _Fake:
        message_text = ""
        media_analysis = None
        business = None
        domain_mismatch = False

    for label, text, want in guards:
        f = _Fake()
        f.message_text = text
        got = bool(scam_signals(f))
        check("%s -> %s" % (label, "flagged" if want else "not flagged"),
              got == want, "signals=%s" % scam_signals(f))

    print("\n4. DISTRIBUTION SHIFT")
    for label, res in (("before", before), ("after", after)):
        da: Dict[str, int] = {}
        dt: Dict[str, int] = {}
        for r in res.values():
            da[r["action"]] = da.get(r["action"], 0) + 1
            dt[r["message_type"]] = dt.get(r["message_type"], 0) + 1
        print("  %-7s action=%s" % (label, dict(sorted(da.items()))))
        print("  %-7s type  =%s" % ("", dict(sorted(dt.items()))))

    if sample:
        print("\n5. ACCURACY vs GOLD (overrides must not make things worse)")
        gold = {r["message_id"]: r for r in ds.sample_messages}
        for label, res in (("before overrides", before), ("after overrides", after)):
            a = sum(1 for m, r in res.items()
                    if m in gold and r["action"] == gold[m]["action"])
            t = sum(1 for m, r in res.items()
                    if m in gold and r["message_type"] == gold[m]["message_type"])
            n = sum(1 for m in res if m in gold)
            print("  %-17s action %d/%d (%.0f%%)   type %d/%d (%.0f%%)"
                  % (label, a, n, 100.0 * a / n, t, n, 100.0 * t / n))
        a_before = sum(1 for m, r in before.items()
                       if m in gold and r["action"] == gold[m]["action"])
        a_after = sum(1 for m, r in after.items()
                      if m in gold and r["action"] == gold[m]["action"])
        check("overrides do not reduce action accuracy on the sample set",
              a_after >= a_before, "before=%d after=%d" % (a_before, a_after))

    print("\n" + "=" * 74)
    if failures:
        print("FAILED - %d failing check(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SAFETY OVERRIDES OK - all checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
