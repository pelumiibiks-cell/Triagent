"""Classification"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from data_layer import Dataset, load_dataset  # noqa: E402
from context_builder import build_all  # noqa: E402
from media_analysis import _call_with_retry, attach_to_contexts, load_cache  # noqa: E402
from evidence import (  # noqa: E402
    EvidenceItem,
    assert_contract,
    retrieve_evidence,
)

# allowed values -- the output contract

ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = ("personal", "urgent", "event", "payment", "business_update",
                 "promotion", "greeting", "forward", "spam", "scam", "unknown")


# tuning constants


# Deterministic pre-filter. Calibrated against the real spread: the 7 scam
# impersonations sit at reports>=20 / age<=34d / unverified, while every
# legitimate sender is verified with a 900+ day account and <=9 reports. The
# gap is enormous, so these thresholds sit in empty space rather than near a
# decision boundary. Verified brands with a mismatched sending domain
# (Thrillophilia, Polaris) are deliberately NOT caught here.
PREFILTER_MIN_REPORTS = 20
PREFILTER_MAX_ACCOUNT_AGE_DAYS = 90

# Candidates shown to the model; it cites at most MAX_CITED of them.
EVIDENCE_CANDIDATES = 5
MAX_CITED = 3

# Below this the first-pass answer is re-asked on the heavier model.
AMBIGUITY_THRESHOLD = 0.60

CHECKPOINT_PATH = os.path.join(config.CACHE_DIR, "classifications.jsonl")
DRYRUN_CHECKPOINT = os.path.join(config.CACHE_DIR, "classifications_sample.jsonl")

API_CALLS = 0



# structured output schema


from pydantic import BaseModel, Field  # noqa: E402


class Routing(BaseModel):
    """Forced output shape. Enum membership is enforced again after parsing."""

    action: str = Field(description="Exactly one of: notify, digest, mute")
    message_type: str = Field(
        description="Exactly one of: personal, urgent, event, payment, "
                    "business_update, promotion, greeting, forward, spam, "
                    "scam, unknown")
    reason: str = Field(
        description="One sentence, 10-20 words, third person, ending in a "
                    "period, explaining WHY this routing decision fits THIS "
                    "user. Never quote the message verbatim.")
    confidence: float = Field(description="Float between 0 and 1.")
    evidence_message_ids: List[str] = Field(
        default_factory=list,
        description="Zero to three message_ids copied EXACTLY from the "
                    "supplied evidence candidates. Empty list if none is "
                    "genuinely relevant.")


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the routing engine for a WhatsApp notification \
system. For each incoming message you decide whether to interrupt the user \
now, batch it for later, or suppress it.

ACTIONS
- notify: important enough to interrupt this user right now.
- digest: useful or harmless, but can wait for a batched summary.
- mute: repetitive, unwanted, low-value, suspicious, scam-like or unsafe.

MESSAGE TYPES (choose the single best fit)
- personal: ordinary one-to-one conversation between people.
- urgent: the message asks this user to DO something soon - act, respond, \
prepare, help, decide - or warns of a disruption they must handle, and the \
subject is NOT money. If the user must take an action shortly, or is directly \
asked a question needing a prompt reply, it is `urgent`.
- event: a scheduled happening or a change to one, conveyed for the user's \
information - meetings, trips, timetables, invitations, schedule shifts. Use \
`event` when the user mainly needs to KNOW; use `urgent` when they must ACT.
- payment: the message is fundamentally about money owed, paid or refunded \
between this user and a party they actually deal with - a bill, invoice, fee, \
dues, maintenance charge, receipt, or a reminder to pay. Choose `payment` \
whenever money changing hands is the SUBJECT of the message.
- business_update: a real transactional or service update from a business the \
user has a relationship with (order, delivery, booking, statement).
- promotion: marketing, offers, discounts, sales, and person-to-person \
selling or marketplace listings. Use this whenever the content is genuinely \
trying to sell or advertise something, even if the user finds it unwanted - \
`mute` is the action for that, `promotion` is still the type.
- greeting: pleasantries, festival wishes, good-morning messages.
- forward: chain/forwarded content passed along rather than written for this \
user.
- spam: unsolicited BULK junk from a stranger or robocaller with no \
legitimate promotional identity behind it. If a real, identifiable seller or \
brand is advertising, prefer `promotion`.
- scam: deliberate fraud - credential/OTP harvesting, payment redirection, \
brand impersonation, too-good-to-be-true offers with urgency.
- unknown: the purpose genuinely cannot be determined, or an unfamiliar \
sender makes a cold approach - an out-of-the-blue enquiry, a wrong number, \
someone who "found your number" somewhere. Prefer `unknown` over `personal` \
whenever the user has no established relationship with the sender, and route \
such cold approaches to `digest` rather than `notify`: an unverified stranger \
asking a question does not earn an interruption.

CHOOSING BETWEEN TYPES WHEN SEVERAL SEEM TO FIT
`action` and `message_type` are INDEPENDENT. Deciding to interrupt the user \
says nothing about which type applies - a payment reminder that must be seen \
today is `notify` + `payment`, never `notify` + `urgent`.

Resolve type overlaps in this order:
1. Is the payment ask fraudulent, or is any fraud signal present? -> `scam`. \
Money plus deception is always `scam`, never `payment`.
2. Is the message fundamentally ABOUT money owed, paid or refunded - a bill, \
invoice, fee, dues, maintenance charge, receipt, or a reminder to pay? -> \
`payment`. This holds even when the message is time-critical and even when \
the action is `notify`. Being urgent does not make it `urgent`.
   This covers the whole payment PROCESS, not just a stated amount: a deadline \
for paying dues or maintenance, instructions on where or how to pay, and \
requests to send, submit or match a receipt or proof of payment are all \
`payment`. A message is still `payment` when it is addressed to the user \
directly or mentions them by name.
3. Is it a business telling the user about the state of a transaction they \
already have - an order, delivery, booking, or an account statement? -> \
`business_update`, even if a sum of money is mentioned in passing. A delivery \
window or a monthly statement is a status update, not a demand for payment.
4. Otherwise apply the definitions above. Reserve `urgent` for non-payment \
calls to action: incidents, emergencies, deadlines, direct requests for help.

Merely MENTIONING money does not make something `payment`. An engineering \
incident about a failing payment service is `urgent`; a colleague asking about \
a refund edge case at work is `urgent` or `personal`. Ask what the message is \
FOR, not which words it contains.

HOW TO DECIDE
Personalise. The same message can deserve different actions for different \
users. Weigh, in order:
1. SAFETY. Clear scam or risk signals mean `mute` with type `scam` (or `spam` \
for junk), regardless of how much the user normally engages with that sender. \
Strong signals: requests for OTP/passwords/codes, payment redirection to a \
new destination, a sending domain that does not match the brand's official \
domain, a very new account with many reports, prize/refund bait plus urgency.
2. THE USER'S OWN HISTORY. The evidence candidates show what this user did \
with similar past messages. If they consistently opened and replied, lean \
`notify`. If they consistently dismissed, ignored, muted or reported, lean \
`digest` or `mute`. This is your strongest personalisation signal.
3. RELATIONSHIP. A business the user actually orders from sending a real \
order update is `business_update` and usually worth `notify` or `digest`. \
The same message from a business they have no relationship with, or have \
opted out of promotions from, leans `digest` or `mute`.
4. GROUPS. A muted group means the user has already said they do not want \
this chatter. Low-value content there - routine greetings, chain forwards, \
repetitive marketplace or promotional posts, anything the user has previously \
dismissed - should be `mute`, not `digest`. Reserve `digest` for group content \
that is genuinely useful to this user later. BUT a genuine direct mention of \
this user, or genuinely urgent content aimed at them, still justifies \
`notify` even in a muted group. Check the direct_mention signal.
   Repetition itself is a mute signal: if the evidence shows the user has \
received near-identical messages before and ignored, dismissed or muted them, \
this one is `mute`.
5. TIMING. If the message arrives inside the user's do-not-disturb window, \
prefer `digest` over `notify` for borderline cases. This is a mild \
tiebreaker only - genuinely urgent messages still notify at night.

MEDIA
Image and voice content has already been extracted for you and appears in \
media_analysis. Treat it as part of the message.
- Voice transcripts come from automatic speech recognition and contain \
recognition errors. Reason about what the speaker MEANS, never about exact \
wording. A garbled or oddly-worded transcript is NOT itself suspicious.
- An ordinary promotional poster is not a scam. Judge images on intent, not \
on the mere presence of a brand name or the word "scam" - a bank's own fraud \
awareness poster is a legitimate business update, not a threat.

EVIDENCE
You are given candidate historical messages, each with the user's actual past \
reaction. Cite only those that genuinely informed your decision - copy their \
message_id EXACTLY. Cite 0 to 3; one is typical. Return an empty list when \
none is genuinely relevant; do not pad. Never invent an id.

REASON STYLE
One sentence, 10-20 words, third person, ending in a period. Explain WHY the \
decision fits THIS user, referring to the signal that drove it. Do not quote \
the message. Examples of the expected register:
- "A trusted group admin sent a time-sensitive update that should interrupt the user."
- "The message is promotional but matches a topic or business the user has opted into."
- "A verified business is sending an update that matches the user's recent order history."
- "The sender's domain does not match the official brand domain, indicating impersonation."

CONFIDENCE
A float in [0,1]. Keep essentially all answers within 0.78 to 0.91. Use \
roughly 0.85-0.91 for clear `notify` calls, 0.81-0.87 for `mute`, and \
0.78-0.84 for routine `digest`. Do not exceed 0.91 - no decision here is \
certain enough to warrant it - and do not drop below 0.78 unless the case is \
genuinely borderline."""


def build_user_prompt(ctx: Any, candidates: Sequence[EvidenceItem]) -> str:
    """Render one message's full context bundle plus its evidence candidates."""
    bundle = ctx.to_prompt_dict()
    bundle.pop("evidence", None)          # candidates are rendered separately
    lines = [
        "Route this message.",
        "",
        "CONTEXT:",
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
        "",
    ]
    if candidates:
        lines.append("EVIDENCE CANDIDATES (cite by message_id, or none):")
        for e in candidates:
            lines.append(json.dumps(e.to_prompt_dict(), ensure_ascii=False,
                                    default=str))
    else:
        lines.append("EVIDENCE CANDIDATES: none available for this user.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# deterministic pre-filter
# ---------------------------------------------------------------------------

def prefilter(ctx: Any) -> Optional[Dict[str, Any]]:
    """Overwhelming scam only. Returns a routing dict, or None to use the LLM.

    Requires ALL of: a mismatched sending domain, an unverified account, a
    brand-new account, and a high recent report count. Any legitimate business
    fails at least one of these.
    """
    biz = ctx.business
    if not biz or not ctx.domain_mismatch:
        return None
    if biz.get("verified"):
        return None
    reports = biz.get("user_reports_30d") or 0
    age = biz.get("account_age_days")
    if reports < PREFILTER_MIN_REPORTS:
        return None
    if age is None or age > PREFILTER_MAX_ACCOUNT_AGE_DAYS:
        return None
    return {
        "action": "mute",
        "message_type": "scam",
        "reason": ("An unverified new account is impersonating this brand from "
                   "a mismatched domain with many recent reports."),
        "confidence": 0.91,
        "evidence_message_ids": [],
        "_source": "prefilter",
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    pass


def validate(result: Dict[str, Any], ctx: Any,
             candidates: Sequence[EvidenceItem]) -> Dict[str, Any]:
    """Enforce the output contract. Raises ValidationError on any breach."""
    action = (result.get("action") or "").strip().lower()
    if action not in ACTIONS:
        raise ValidationError("action %r not in %s" % (result.get("action"), ACTIONS))

    mtype = (result.get("message_type") or "").strip().lower()
    if mtype not in MESSAGE_TYPES:
        raise ValidationError("message_type %r not in allowed set"
                              % result.get("message_type"))

    reason = (result.get("reason") or "").strip()
    if not reason:
        raise ValidationError("reason is empty")
    reason = " ".join(reason.split())     # collapse newlines for CSV safety

    try:
        conf = float(result.get("confidence"))
    except (TypeError, ValueError):
        raise ValidationError("confidence %r is not a float"
                              % result.get("confidence")) from None
    if not 0.0 <= conf <= 1.0:
        raise ValidationError("confidence %r outside [0,1]" % conf)

    allowed = {e.message_id for e in candidates}
    ids: List[str] = []
    for raw in result.get("evidence_message_ids") or []:
        mid = str(raw).strip()
        if not mid or mid.lower() == "none":
            continue
        if mid not in allowed:
            raise ValidationError(
                "evidence id %r was not among the supplied candidates %s"
                % (mid, sorted(allowed)))
        if mid not in ids:
            ids.append(mid)
    if len(ids) > MAX_CITED:
        ids = ids[:MAX_CITED]

    return {
        "message_id": ctx.message_id,
        "action": action,
        "message_type": mtype,
        "reason": reason,
        "confidence": round(conf, 2),
        "evidence_message_ids": ids,
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def classify_one(client, ctx: Any, candidates: Sequence[EvidenceItem],
                 model: str) -> Dict[str, Any]:
    """One structured call. Returns the raw parsed dict."""
    global API_CALLS
    from google.genai import types

    def _do():
        return client.models.generate_content(
            model=model,
            contents=build_user_prompt(ctx, candidates),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=config.TEMPERATURE,
                response_mime_type="application/json",
                response_schema=Routing,
            ),
        )

    resp = _call_with_retry(_do, ctx.message_id)
    API_CALLS += 1
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        raise ValidationError("unparseable model output: %r"
                              % getattr(resp, "text", None))
    return parsed.model_dump()


def route_message(client, ds: Dataset, ctx: Any, media_cache: Dict[str, Any],
                  bulk_model: str, heavy_model: str) -> Dict[str, Any]:
    """Full decision for one message: pre-filter, classify, validate, escalate."""
    candidates = retrieve_evidence(ds, ctx, k=EVIDENCE_CANDIDATES,
                                   media_cache=media_cache)

    pre = prefilter(ctx)
    if pre is not None:
        out = validate(pre, ctx, candidates)
        out["_source"] = "prefilter"
        out["_model"] = None
        return out

    attempts: List[str] = []
    for model in (bulk_model, heavy_model):
        try:
            raw = classify_one(client, ctx, candidates, model)
            out = validate(raw, ctx, candidates)
            # a very low-confidence answer from the cheap model is escalated
            if model is bulk_model and out["confidence"] < AMBIGUITY_THRESHOLD:
                attempts.append("%s: low confidence %.2f"
                                % (model, out["confidence"]))
                continue
            out["_source"] = "llm"
            out["_model"] = model
            if attempts:
                out["_retries"] = attempts
            return out
        except ValidationError as exc:
            attempts.append("%s: %s" % (model, exc))
            continue

    raise RuntimeError("classification failed for %s after %d attempts: %s"
                       % (ctx.message_id, len(attempts), attempts))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> Dict[str, Dict[str, Any]]:
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
                continue              # tolerate a torn final line after a crash
            if rec.get("message_id"):
                out[rec["message_id"]] = rec
    return out


def append_checkpoint(path: str, record: Dict[str, Any]) -> None:
    config.ensure_cache_dir()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# batch run
# ---------------------------------------------------------------------------

def run(ds: Optional[Dataset] = None, sample: bool = False,
        limit: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
    ds = ds or load_dataset()
    media_cache = load_cache()

    rows = ds.sample_messages if sample else ds.messages
    ckpt_path = DRYRUN_CHECKPOINT if sample else CHECKPOINT_PATH
    contexts = build_all(ds, rows)
    attach_to_contexts(contexts, media_cache)

    done = {} if force else load_checkpoint(ckpt_path)
    todo = [c for c in contexts if c.message_id not in done]
    if limit:
        todo = todo[:limit]

    print("%d messages, %d already checkpointed, %d to classify"
          % (len(contexts), len(contexts) - len([c for c in contexts
                                                 if c.message_id not in done]),
             len(todo)))

    client = None
    bulk = config.MODEL_BULK
    heavy = config.MODEL_HEAVY
    if todo:
        client = config.get_client()
        bulk, heavy, _ = config.verify_models(client, verbose=True)
        print("  bulk=%s  heavy=%s" % (bulk, heavy))

    errors: List[Dict[str, str]] = []
    t0 = time.time()
    for i, ctx in enumerate(todo, 1):
        try:
            rec = route_message(client, ds, ctx, media_cache, bulk, heavy)
            append_checkpoint(ckpt_path, rec)
            done[ctx.message_id] = rec
            if i % 10 == 0 or i == len(todo):
                print("  [%3d/%3d] %-16s %-7s %-16s conf=%.2f  (%.0fs elapsed)"
                      % (i, len(todo), ctx.message_id, rec["action"],
                         rec["message_type"], rec["confidence"], time.time() - t0))
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors.append({"message_id": ctx.message_id,
                           "error": "%s: %s" % (type(exc).__name__, exc)})
            print("  [%3d/%3d] %-16s FAILED - %s"
                  % (i, len(todo), ctx.message_id, exc))

    return {"results": done, "contexts": contexts, "errors": errors,
            "api_calls": API_CALLS, "elapsed": time.time() - t0,
            "checkpoint": ckpt_path}


# ---------------------------------------------------------------------------
# dry-run scoring against the sample gold labels
# ---------------------------------------------------------------------------

def score_against_gold(ds: Dataset, results: Dict[str, Dict[str, Any]]) -> None:
    gold = {r["message_id"]: r for r in ds.sample_messages}
    scored = [(m, r) for m, r in results.items() if m in gold]
    if not scored:
        print("no sample rows to score")
        return

    a_ok = sum(1 for m, r in scored if r["action"] == gold[m]["action"])
    t_ok = sum(1 for m, r in scored if r["message_type"] == gold[m]["message_type"])
    both = sum(1 for m, r in scored
               if r["action"] == gold[m]["action"]
               and r["message_type"] == gold[m]["message_type"])
    n = len(scored)
    print("\n  action accuracy      : %d/%d (%.0f%%)" % (a_ok, n, 100.0 * a_ok / n))
    print("  message_type accuracy: %d/%d (%.0f%%)" % (t_ok, n, 100.0 * t_ok / n))
    print("  both correct         : %d/%d (%.0f%%)" % (both, n, 100.0 * both / n))

    confs = [r["confidence"] for _, r in scored]
    print("  confidence: min=%.2f max=%.2f mean=%.3f (sample gold band 0.78-0.91)"
          % (min(confs), max(confs), sum(confs) / len(confs)))
    wl = [len(r["reason"].split()) for _, r in scored]
    print("  reason words: min=%d max=%d mean=%.1f (gold 10-20, mean 13.7)"
          % (min(wl), max(wl), sum(wl) / len(wl)))

    print("\n  action confusion (gold -> predicted):")
    conf: Dict[str, Dict[str, int]] = {}
    for m, r in scored:
        g = gold[m]["action"]
        conf.setdefault(g, {})
        conf[g][r["action"]] = conf[g].get(r["action"], 0) + 1
    for g in sorted(conf):
        print("    %-8s -> %s" % (g, dict(sorted(conf[g].items()))))

    print("\n  disagreements:")
    for m, r in sorted(scored):
        g = gold[m]
        if r["action"] == g["action"] and r["message_type"] == g["message_type"]:
            continue
        print("    %-16s gold=%-7s/%-16s got=%-7s/%-16s"
              % (m, g["action"], g["message_type"], r["action"], r["message_type"]))
        print("        reason: %s" % r["reason"])


def _main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    sample = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    limit = None
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    print("=" * 74)
    print("PHASE 5 - CLASSIFICATION%s" % ("  [DRY RUN vs sample gold]" if sample else ""))
    print("=" * 74)

    ds = load_dataset()

    if "--stats" in sys.argv:
        for label, path, n in (("full batch", CHECKPOINT_PATH, len(ds.messages)),
                               ("dry run", DRYRUN_CHECKPOINT,
                                len(ds.sample_messages))):
            ck = load_checkpoint(path)
            print("  %-12s %3d/%3d checkpointed at %s" % (label, len(ck), n, path))
        return 0

    # how many messages would the pre-filter catch, for the record
    ctxs = build_all(ds, ds.sample_messages if sample else ds.messages)
    caught = [c.message_id for c in ctxs if prefilter(c)]
    print("pre-filter catches %d/%d messages deterministically: %s"
          % (len(caught), len(ctxs), caught))
    print("remaining need an LLM call: %d" % (len(ctxs) - len(caught)))

    res = run(ds, sample=sample, limit=limit, force=force)
    results = res["results"]

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  classified        : %d" % len(results))
    print("  GEMINI API CALLS  : %d" % res["api_calls"])
    print("  wall clock        : %.1fs" % res["elapsed"])
    print("  errors            : %d" % len(res["errors"]))
    for e in res["errors"]:
        print("      %s: %s" % (e["message_id"], e["error"]))
    print("  checkpoint        : %s" % res["checkpoint"])

    src: Dict[str, int] = {}
    for r in results.values():
        src[r.get("_source", "?")] = src.get(r.get("_source", "?"), 0) + 1
    print("  decision source   : %s" % dict(sorted(src.items())))
    esc = [m for m, r in results.items() if r.get("_retries")]
    print("  escalated to heavy: %d %s" % (len(esc), esc[:6]))

    dist_a: Dict[str, int] = {}
    dist_t: Dict[str, int] = {}
    for r in results.values():
        dist_a[r["action"]] = dist_a.get(r["action"], 0) + 1
        dist_t[r["message_type"]] = dist_t.get(r["message_type"], 0) + 1
    print("\n  action distribution      : %s" % dict(sorted(dist_a.items())))
    print("  message_type distribution: %s" % dict(sorted(dist_t.items())))

    bad_a = [r for r in results.values() if r["action"] not in ACTIONS]
    bad_t = [r for r in results.values() if r["message_type"] not in MESSAGE_TYPES]
    bad_c = [r for r in results.values()
             if not 0.0 <= float(r["confidence"]) <= 1.0]
    print("\n  enum violations: action=%d message_type=%d confidence=%d"
          % (len(bad_a), len(bad_t), len(bad_c)))

    ncited = sum(len(r["evidence_message_ids"]) for r in results.values())
    nnone = sum(1 for r in results.values() if not r["evidence_message_ids"])
    print("  evidence: %d ids cited, %d rows cite none" % (ncited, nnone))

    if sample:
        print("\n" + "=" * 74)
        print("DRY-RUN SCORING vs sample_messages.csv GOLD")
        print("=" * 74)
        score_against_gold(ds, results)

    print("\n  example decisions:")
    for r in list(results.values())[:5]:
        print("    %-16s %-7s %-16s conf=%.2f ev=%s"
              % (r["message_id"], r["action"], r["message_type"],
                 r["confidence"], r["evidence_message_ids"] or "none"))
        print("        %s" % r["reason"])

    return 1 if res["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
