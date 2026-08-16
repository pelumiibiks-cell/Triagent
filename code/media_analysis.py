"""Phase 3 - Media understanding.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from data_layer import Dataset, load_dataset  # noqa: E402

# API call accounting, so the run can report real usage.
API_CALLS = 0
_last_call_ts = 0.0


# ---------------------------------------------------------------------------
# structured output schema
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    raise SystemExit("pydantic is required (it ships with google-genai). "
                     "Run: pip install google-genai")


class ImageAnalysis(BaseModel):
    """Forced shape for the vision extraction. No free-text parsing."""

    visible_text: str = Field(
        description="ALL text visible in the image, transcribed verbatim, "
                    "preserving line breaks. Empty string if there is none.")
    apparent_intent: str = Field(
        description="What this image is trying to get the viewer to do or know, "
                    "in one sentence.")
    red_flags: List[str] = Field(
        default_factory=list,
        description="Risk signals present: urgency/pressure language, "
                    "suspicious or shortened links, payment or QR requests, "
                    "brand impersonation, credential/OTP requests, "
                    "too-good-to-be-true offers. Empty list if none.")
    has_payment_request: bool = Field(
        description="True if the image asks for money, a payment, a fee, or "
                    "bank/UPI/wallet details.")
    has_qr_code: bool = Field(
        description="True if a QR code is visibly present in the image.")
    has_link: bool = Field(
        description="True if any URL, domain, or web link is visible.")
    brand_mentioned: Optional[str] = Field(
        default=None,
        description="The brand or organisation the image presents itself as, "
                    "or null if none is claimed.")
    summary: str = Field(
        description="One short line describing the image for a notification "
                    "routing decision.")


IMAGE_PROMPT = """You are analysing an image attached to a WhatsApp message, \
for a notification-routing system that must decide whether to interrupt the \
user, batch the message into a digest, or mute it.

Extract exactly what is in the image. Do not speculate beyond what is visible.

- visible_text: transcribe ALL readable text verbatim, preserving line breaks.
- apparent_intent: one sentence on what the image wants the viewer to do or know.
- red_flags: list only signals that are actually present. Consider urgency or \
pressure language, suspicious/shortened/lookalike links, payment or QR \
requests, brand impersonation, credential or OTP requests, and \
too-good-to-be-true offers.
- has_payment_request / has_qr_code / has_link: strictly what is visible.
- brand_mentioned: the brand the image presents itself as, or null.
- summary: one short line useful for a routing decision.

Be precise and literal. An ordinary promotional poster is NOT a red flag by \
itself; only flag genuine risk signals."""


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def load_cache(path: str = config.MEDIA_ANALYSIS_CACHE) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print("  ! cache unreadable (%s) - starting a fresh cache" % exc)
        return {}


def save_cache(cache: Dict[str, Any],
               path: str = config.MEDIA_ANALYSIS_CACHE) -> None:
    """Write the whole cache atomically. Called after EVERY item."""
    config.ensure_cache_dir()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# rate limiting / retry
# ---------------------------------------------------------------------------

def _pace() -> None:
    """Sleep so we stay under the per-minute cap instead of bursting."""
    global _last_call_ts
    if _last_call_ts:
        wait = config.MIN_REQUEST_INTERVAL - (time.time() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
    _last_call_ts = time.time()


def _is_retryable(exc: Exception) -> bool:
    """429 rate limits plus transient server-side failures.

    The first Phase 3 run lost img_008 to a 503 'Deadline expired' -- a
    transient server error, not a quota problem. Retrying only on 429 would
    keep dropping those, so 503/UNAVAILABLE/deadline/timeout are retryable too.
    """
    s = ("%s %s" % (type(exc).__name__, exc)).lower()
    rate_limited = ("429" in s or "resource_exhausted" in s
                    or "rate limit" in s or "quota" in s)
    transient = ("503" in s or "unavailable" in s or "deadline" in s
                 or "500" in s or "internal error" in s or "timeout" in s)
    return rate_limited or transient


def _call_with_retry(fn, label: str):
    """Run fn(), retrying with exponential backoff on 429/RESOURCE_EXHAUSTED."""
    global API_CALLS
    backoff = config.INITIAL_BACKOFF_SECONDS
    last: Optional[Exception] = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            _pace()
            API_CALLS += 1
            return fn()
        except Exception as exc:  # noqa: BLE001 - must classify then re-raise
            last = exc
            if not _is_retryable(exc):
                raise
            if attempt == config.MAX_RETRIES:
                break
            wait = min(backoff, config.MAX_BACKOFF_SECONDS)
            print("    retryable error on %s (attempt %d/%d: %s) - backing off %.0fs"
                  % (label, attempt, config.MAX_RETRIES,
                     type(exc).__name__, wait))
            time.sleep(wait)
            backoff *= config.BACKOFF_MULTIPLIER
    raise RuntimeError("giving up on %s after %d rate-limited attempts: %s"
                       % (label, config.MAX_RETRIES, last))


# ---------------------------------------------------------------------------
# image analysis
# ---------------------------------------------------------------------------

def analyze_image(client, model: str, path: str, media_id: str) -> Dict[str, Any]:
    """One structured vision call. Raises on non-rate-limit failure."""
    from google.genai import types

    with open(path, "rb") as fh:
        data = fh.read()

    def _do():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                IMAGE_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=config.TEMPERATURE,
                response_mime_type="application/json",
                response_schema=ImageAnalysis,
            ),
        )

    resp = _call_with_retry(_do, media_id)
    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        # schema violation: surface it rather than silently keeping junk
        raise ValueError("model returned unparseable output for %s: %r"
                         % (media_id, getattr(resp, "text", None)))
    out = parsed.model_dump()
    out["kind"] = "image"
    out["media_id"] = media_id
    out["model"] = model
    out["bytes"] = len(data)
    return out


# ---------------------------------------------------------------------------
# voice transcription (local, no API)
# ---------------------------------------------------------------------------

_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("  loading faster-whisper '%s' (first run downloads weights) ..."
              % config.WHISPER_MODEL)
        t0 = time.time()
        _whisper = WhisperModel(config.WHISPER_MODEL, device="cpu",
                                compute_type="int8")
        print("  model ready in %.1fs" % (time.time() - t0))
    return _whisper


def transcribe_voice(path: str, media_id: str) -> Dict[str, Any]:
    """Local ASR. No Gemini quota consumed."""
    model = _get_whisper()
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "kind": "voice",
        "media_id": media_id,
        "transcript": text,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration_seconds": round(float(info.duration), 2),
        "model": "faster-whisper:%s" % config.WHISPER_MODEL,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def required_media(ds: Dataset) -> Dict[str, List[str]]:
    """Distinct media ids any phase will need, across all three message tables."""
    images, voices = set(), set()
    for rows in (ds.messages, ds.sample_messages, ds.history_rows):
        for r in rows:
            if not r["media_id"]:
                continue
            if r["media_type"] == "image":
                images.add(r["media_id"])
            elif r["media_type"] == "voice":
                voices.add(r["media_id"])
    return {"images": sorted(images), "voices": sorted(voices)}


def run(ds: Optional[Dataset] = None, force: bool = False) -> Dict[str, Any]:
    """Analyse all required media, using and updating the cache."""
    ds = ds or load_dataset()
    need = required_media(ds)
    cache = {} if force else load_cache()

    stats = {"image_analyzed": 0, "image_cached": 0,
             "voice_transcribed": 0, "voice_cached": 0,
             "missing_file": 0, "errors": []}

    client = None
    bulk_model = config.MODEL_BULK

    # -- images ------------------------------------------------------------
    todo = [i for i in need["images"] if force or i not in cache
            or "error" in cache.get(i, {})]
    print("images : %d required, %d already cached, %d to analyse"
          % (len(need["images"]), len(need["images"]) - len(todo), len(todo)))
    stats["image_cached"] = len(need["images"]) - len(todo)

    if todo:
        client = config.get_client()
        bulk_model, _, _ = config.verify_models(client, verbose=True)
        print("  using model: %s" % bulk_model)

    for mid in todo:
        path = ds.media_path(mid, "image")
        if not path:
            stats["missing_file"] += 1
            cache[mid] = {"kind": "image", "media_id": mid,
                          "error": "media file missing or unresolvable"}
            save_cache(cache)
            print("  %-10s SKIP - file missing (recorded, run continues)" % mid)
            continue
        try:
            t0 = time.time()
            cache[mid] = analyze_image(client, bulk_model, path, mid)
            save_cache(cache)          # incremental: after EVERY item
            stats["image_analyzed"] += 1
            print("  %-10s ok  (%.1fs) flags=%s"
                  % (mid, time.time() - t0, cache[mid]["red_flags"] or "-"))
        except Exception as exc:  # noqa: BLE001 - degrade, don't kill the run
            msg = "%s: %s" % (type(exc).__name__, exc)
            stats["errors"].append({"media_id": mid, "error": msg})
            cache[mid] = {"kind": "image", "media_id": mid, "error": msg}
            save_cache(cache)
            print("  %-10s FAILED - %s" % (mid, msg))

    # -- voice notes -------------------------------------------------------
    vtodo = [v for v in need["voices"] if force or v not in cache
             or "error" in cache.get(v, {})]
    print("\nvoice  : %d required, %d already cached, %d to transcribe"
          % (len(need["voices"]), len(need["voices"]) - len(vtodo), len(vtodo)))
    stats["voice_cached"] = len(need["voices"]) - len(vtodo)

    for mid in vtodo:
        path = ds.media_path(mid, "voice")
        if not path:
            stats["missing_file"] += 1
            cache[mid] = {"kind": "voice", "media_id": mid,
                          "error": "media file missing or unresolvable"}
            save_cache(cache)
            print("  %-10s SKIP - file missing (recorded, run continues)" % mid)
            continue
        try:
            t0 = time.time()
            cache[mid] = transcribe_voice(path, mid)
            save_cache(cache)
            stats["voice_transcribed"] += 1
            print("  %-10s ok  (%.1fs, %.1fs audio, lang=%s)"
                  % (mid, time.time() - t0, cache[mid]["duration_seconds"],
                     cache[mid]["language"]))
        except Exception as exc:  # noqa: BLE001
            msg = "%s: %s" % (type(exc).__name__, exc)
            stats["errors"].append({"media_id": mid, "error": msg})
            cache[mid] = {"kind": "voice", "media_id": mid, "error": msg}
            save_cache(cache)
            print("  %-10s FAILED - %s" % (mid, msg))

    stats["api_calls"] = API_CALLS
    return {"cache": cache, "stats": stats, "required": need}


def get_analysis(media_id: Optional[str],
                 cache: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Lookup helper for later phases. None when absent or errored."""
    if not media_id:
        return None
    cache = load_cache() if cache is None else cache
    rec = cache.get(media_id)
    if rec is None or "error" in rec:
        return None
    return rec


def attach_to_contexts(contexts, cache: Optional[Dict[str, Any]] = None) -> int:
    """Fill the Phase 2 `media_analysis` slot on each bundle. Returns count."""
    cache = load_cache() if cache is None else cache
    n = 0
    for ctx in contexts:
        rec = get_analysis(ctx.media_id, cache)
        if rec is not None:
            ctx.media_analysis = rec
            n += 1
    return n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _main() -> int:
    # OCR legitimately returns non-cp1252 glyphs (e.g. the rupee sign), which
    # would crash printing on a default Windows console. Force UTF-8 output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    stats_only = "--stats" in sys.argv
    force = "--force" in sys.argv

    print("=" * 74)
    print("PHASE 3 - MEDIA ANALYSIS")
    print("=" * 74)

    ds = load_dataset()
    need = required_media(ds)
    print("required media: %d distinct images, %d distinct voice notes"
          % (len(need["images"]), len(need["voices"])))
    print("  (union across messages.csv, sample_messages.csv, message_history.csv)")

    if stats_only:
        cache = load_cache()
        done = [k for k, v in cache.items() if "error" not in v]
        err = [k for k, v in cache.items() if "error" in v]
        print("\ncache: %d entries (%d ok, %d errored) at %s"
              % (len(cache), len(done), len(err), config.MEDIA_ANALYSIS_CACHE))
        missing = [m for m in need["images"] + need["voices"] if m not in cache]
        print("uncached: %d %s" % (len(missing), missing[:10]))
        return 0

    t0 = time.time()
    result = run(ds, force=force)
    elapsed = time.time() - t0
    cache, stats = result["cache"], result["stats"]

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  images analysed this run  : %d" % stats["image_analyzed"])
    print("  images served from cache  : %d" % stats["image_cached"])
    print("  voice transcribed this run: %d" % stats["voice_transcribed"])
    print("  voice served from cache   : %d" % stats["voice_cached"])
    print("  missing media files       : %d" % stats["missing_file"])
    print("  GEMINI API CALLS          : %d" % stats["api_calls"])
    print("  wall clock                : %.1fs" % elapsed)
    print("  errors                    : %d" % len(stats["errors"]))
    for e in stats["errors"]:
        print("      %s: %s" % (e["media_id"], e["error"]))

    ok_imgs = [v for v in cache.values() if v.get("kind") == "image"
               and "error" not in v]
    ok_vns = [v for v in cache.values() if v.get("kind") == "voice"
              and "error" not in v]
    print("\n  cache now holds %d image analyses, %d transcripts"
          % (len(ok_imgs), len(ok_vns)))
    flagged = [v for v in ok_imgs if v["red_flags"]]
    print("  images with >=1 red flag  : %d/%d" % (len(flagged), len(ok_imgs)))
    print("  images with payment req   : %d" % sum(1 for v in ok_imgs
                                                   if v["has_payment_request"]))
    print("  images with a QR code     : %d" % sum(1 for v in ok_imgs
                                                   if v["has_qr_code"]))
    print("  images with a link        : %d" % sum(1 for v in ok_imgs
                                                   if v["has_link"]))

    # ---- verbatim examples -------------------------------------------------
    print("\n" + "=" * 74)
    print("EXAMPLE RESULTS (verbatim, not summarised)")
    print("=" * 74)

    def show(rec):
        print("\n--- %s ---" % rec["media_id"])
        print(json.dumps(rec, indent=2, ensure_ascii=False))

    scammy = sorted(ok_imgs, key=lambda v: (len(v["red_flags"]),
                                            v["has_payment_request"]),
                    reverse=True)
    clean = [v for v in ok_imgs if not v["red_flags"]]
    if scammy:
        print("\n>>> highest-risk image:")
        show(scammy[0])
    promo = next((v for v in ok_imgs
                  if "promo" in (v["apparent_intent"] or "").lower()
                  or "offer" in (v["visible_text"] or "").lower()
                  or "sale" in (v["visible_text"] or "").lower()), None)
    if promo:
        print("\n>>> promotional-looking image:")
        show(promo)
    elif clean:
        print("\n>>> low-risk image:")
        show(clean[0])
    if len(scammy) > 1:
        print("\n>>> second-highest-risk image:")
        show(scammy[1])
    if ok_vns:
        longest = max(ok_vns, key=lambda v: v["duration_seconds"])
        print("\n>>> voice transcript (longest clip):")
        show(longest)
        print("\n>>> voice transcript (another):")
        show(next(v for v in ok_vns if v["media_id"] != longest["media_id"]))

    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
