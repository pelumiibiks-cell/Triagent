

from __future__ import annotations

import os


# paths


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
MEDIA_DIR = os.path.join(DATASET_DIR, "media")
CACHE_DIR = os.path.join(ROOT, "cache")

MEDIA_ANALYSIS_CACHE = os.path.join(CACHE_DIR, "media_analysis.json")

OUTPUT_CSV = os.path.join(ROOT, "output.csv")


# models

#
# Verified against client.models.list() at setup time -- see
# `verify_models()` below and the Phase 3 report. Do NOT edit these from
# memory; re-run the verifier if you suspect drift.

MODEL_BULK = "gemini-3.5-flash-lite"    # bulk classification + image extraction
MODEL_HEAVY = "gemini-3.5-flash"        # reserved for genuinely ambiguous cases

# Used only if the targets above are absent from the live model list.
MODEL_BULK_FALLBACK = "gemini-2.5-flash-lite"
MODEL_HEAVY_FALLBACK = "gemini-2.5-flash"

# Local ASR. 'base' is enough for these short, clear clips; bump to 'small'
# only if transcripts are visibly poor. No API cost either way.
WHISPER_MODEL = "base"

# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------
#
# Free tier is roughly single-to-low-double-digit RPM. We pace to a
# conservative interval rather than bursting and relying on backoff.

REQUESTS_PER_MINUTE = 10          # conservative; pacing target, not a cap we probe
MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE
MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 5.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 120.0

TEMPERATURE = 0.0                 # repeatable classification


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

ENV_KEY_NAME = "GEMINI_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised at startup when the API key is not available."""


def load_env() -> None:
    """Populate os.environ from the repo-root .env, if python-dotenv is present."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise MissingAPIKeyError(
            "python-dotenv is not installed. Run:\n"
            "    pip install python-dotenv\n"
            "or export %s manually." % ENV_KEY_NAME
        ) from exc
    load_dotenv(os.path.join(ROOT, ".env"))


def get_api_key() -> str:
    """Return the API key, failing loudly and immediately if it is absent.

    Never logs or returns any part of the value in an error message.
    """
    load_env()
    key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not key:
        raise MissingAPIKeyError(
            "Required environment variable %s is not set.\n"
            "Create a .env file in the repo root containing:\n"
            "    %s=your_key_here\n"
            "Get a free key at https://aistudio.google.com/apikey .\n"
            "Never commit this file -- .env is already in .gitignore."
            % (ENV_KEY_NAME, ENV_KEY_NAME)
        )
    return key


def get_client():
    """Construct a google-genai client, failing loudly if the key is missing."""
    key = get_api_key()          # raises with a clear message before any network use
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run:\n"
            "    pip install google-genai\n"
            "(Do NOT install google-generativeai -- that library is deprecated.)"
        ) from exc
    return genai.Client(api_key=key)


def ensure_cache_dir() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


# ---------------------------------------------------------------------------
# model verification
# ---------------------------------------------------------------------------

def list_models(client=None) -> list:
    """Live model ids that support generateContent, from the API itself."""
    client = client or get_client()
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append(m.name.replace("models/", ""))
    return sorted(set(names))


def verify_models(client=None, verbose: bool = True):
    """Confirm the configured model ids exist live; fall back if not.

    Returns (bulk_model, heavy_model, available_list).
    """
    available = list_models(client)
    bulk, heavy = MODEL_BULK, MODEL_HEAVY
    if bulk not in available:
        if verbose:
            print("  ! %s not in live model list - falling back to %s"
                  % (bulk, MODEL_BULK_FALLBACK))
        bulk = MODEL_BULK_FALLBACK
    if heavy not in available:
        if verbose:
            print("  ! %s not in live model list - falling back to %s"
                  % (heavy, MODEL_HEAVY_FALLBACK))
        heavy = MODEL_HEAVY_FALLBACK
    for name, chosen in (("bulk", bulk), ("heavy", heavy)):
        if chosen not in available:
            raise RuntimeError(
                "Neither the target nor the fallback %s model is available. "
                "Live list: %s" % (name, available)
            )
    return bulk, heavy, available


if __name__ == "__main__":
    print("ROOT        :", ROOT)
    print("DATASET_DIR :", DATASET_DIR)
    print("CACHE_DIR   :", CACHE_DIR)
    print("cache file  :", MEDIA_ANALYSIS_CACHE)
    try:
        get_api_key()
        print("%s          : present (value not shown)" % ENV_KEY_NAME)
    except MissingAPIKeyError as exc:
        print("%s          : MISSING\n%s" % (ENV_KEY_NAME, exc))
        raise SystemExit(1)
    print("\nverifying model ids against the live API ...")
    bulk, heavy, available = verify_models()
    print("  bulk  ->", bulk)
    print("  heavy ->", heavy)
    print("\n%d models support generateContent. flash/flash-lite/pro family:"
          % len(available))
    for n in available:
        if any(k in n for k in ("flash", "pro")) and "vision" not in n:
            print("   ", n)
