"""Generate pre-labelled synthetic patient messages with Claude Haiku 5.5.

Approach (rewrite + pre-label, then human review):
  1. Seeds are REAL patient messages from the healthcare-chatbot-intent data, taken
     only from the intent model's train/val split. The 437 test-split messages are
     never used as seeds, so the real-data test set stays unseen.
  2. GENERATE (call 1): for each seed, Claude rewrites the message three times -
     negative, neutral and positive tone - keeping the same request/intent. A
     deterministic style (length, register, quirk) is assigned per job for diversity.
  3. LABEL (call 2, blind): a separate call labels each rewrite (intent, sentiment,
     emergency flag) WITHOUT seeing the target sentiment, so the label is an
     independent check rather than an echo of the instruction.
  4. Action = action_policy table(label intent, label sentiment); a message the
     labeller flags as a medical emergency becomes route_to_human_emergency.
  5. Rows are flagged `needs_review` when the blind label disagrees with the target
     sentiment or the seed intent, or when an emergency is flagged. Human reviewers
     fill reviewer_* columns; training uses reviewer labels where present.

Repeatability: the job plan (which seeds, target counts, styles) is fully
determined by --seed. Every API response is cached in a JSONL file keyed by job id,
so re-running resumes and never re-bills finished jobs; the cached file is the
reproducible artifact (LLM outputs themselves are not bit-for-bit deterministic).

Usage:
    python generate_synthetic.py --dry-run                 # plan + cost estimate, no API calls
    python generate_synthetic.py --limit 20                # pilot: first 20 jobs (~60 messages)
    python generate_synthetic.py                           # full run (~3,334 jobs -> 10,000 messages)
    python generate_synthetic.py --build-only              # rebuild the CSV from the cache

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import action_policy as policy

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "synthetic"
CACHE_JSONL = OUT_DIR / "generations.jsonl"
OUT_CSV = OUT_DIR / "healthcare_synthetic.csv"

MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"
TARGET_MESSAGES = 10_000
SENTIMENTS = list(policy.SENTIMENTS)
MAX_JOBS_PER_SEED = 12  # caps how often a rare-intent seed is reused
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}  # claude-haiku-5-5, prompts < 100K tokens

INTENT_DESCRIPTIONS = {
    "emergency": "a medical emergency needing immediate care (see emergency definition)",
    "book_appointment": "book, schedule, confirm or choose a time for an appointment, or ask about availability",
    "reschedule_appointment": "move an existing appointment to a different time",
    "cancel_appointment": "cancel an existing appointment",
    "insurance": "insurance coverage, claims, copays, benefits, eligibility, policy/plan details, deductibles",
    "payment": "bills, paying, payment methods, prices or costs of services",
    "clinic_info": "clinic or hospital hours, locations, contact details, portal or practice information",
    "find_provider": "find a doctor, specialist, pharmacy or facility",
    "provide_information": "giving requested personal or verification details (name, DOB, IDs, preferences)",
    "symptom_check": "describing or asking about symptoms (non-emergency)",
    "referral": "request or ask about a referral or second opinion",
    "general_health_info": "general health, condition, vaccine, procedure or wellness questions",
    "account_update": "update profile, contact or account details",
    "medication_info": "questions about a medication's use, dosage, side effects or interactions",
    "prescription_refill": "refill, renew, transfer or check the status of a prescription",
    "lab_results": "ask about or request lab/test results",
    "other": "anything else, including medical records requests",
}
INTENTS = sorted(policy.INTENT_GROUP)
assert set(INTENTS) == set(INTENT_DESCRIPTIONS), "INTENT_DESCRIPTIONS must match action_policy intents"
IntentLit = Literal[tuple(INTENTS)]  # type: ignore[valid-type]
SentimentLit = Literal["negative", "neutral", "positive"]

SENTIMENT_DEFINITIONS = """\
- negative: the patient sounds frustrated, upset, angry, anxious, distressed or is complaining \
(e.g. a long wait, a billing error, worsening or worrying symptoms described with distress).
- neutral: a matter-of-fact request, question or piece of information with no clear emotion.
- positive: the patient sounds grateful, pleased, relieved, satisfied or enthusiastic."""

EMERGENCY_DEFINITION = """\
A medical emergency is a message describing something that may need immediate care: chest pain or \
pressure, difficulty breathing, fainting/collapse/unresponsiveness, stroke signs (face droop, slurred \
speech, one-sided weakness), severe bleeding, severe allergic reaction, seizure, suicidal thoughts or \
self-harm, overdose or poisoning, or severe sudden pain. Routine or mild symptoms are NOT emergencies."""

STYLES = {
    "length": ["very short (under 8 words)", "one sentence", "two or three sentences"],
    "register": ["casual", "formal", "terse", "chatty", "worried"],
    "quirk": ["none", "none", "a small typo", "no punctuation or capitals", "common abbreviations",
              "slightly non-native English"],
}

GENERATE_SYSTEM = f"""You write realistic patient messages sent to a healthcare practice's chat assistant, \
for training a message-routing classifier. All names, IDs and numbers you write must be fictional.

Given one real patient message and its intent, write THREE new messages with the SAME intent (the same \
kind of request), one for each tone: negative, neutral, positive.

Tone definitions:
{SENTIMENT_DEFINITIONS}

Rules:
- Keep the intent identical to the original; vary wording, details, names, dates and specifics.
- Do not copy the original sentence; each message must read as a different person writing.
- Follow the requested style (length, register, quirk) for all three messages.
- For the emergency intent, every version must still clearly describe an emergency; tone only changes \
how the person writes (e.g. polite and calm vs. panicked).
- Write only the patient's message, no labels or commentary."""

LABEL_SYSTEM = f"""You label patient messages sent to a healthcare practice's chat assistant.

For each message, give:
- intent: exactly one of the intents below (the patient's main request).
- sentiment: negative, neutral or positive.
- is_emergency: true only if the message meets the emergency definition.
- rationale: one short sentence.

Intents:
{chr(10).join(f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())}

Sentiment definitions:
{SENTIMENT_DEFINITIONS}

Emergency definition:
{EMERGENCY_DEFINITION}

Label each message on its own text only."""

log = logging.getLogger("generate_synthetic")


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #
class Variant(BaseModel):
    target_sentiment: SentimentLit
    text: str


class Generation(BaseModel):
    variants: list[Variant] = Field(description="Exactly three messages: negative, neutral, positive")


class Label(BaseModel):
    index: int
    intent: IntentLit
    sentiment: SentimentLit
    is_emergency: bool
    rationale: str


class LabelBatch(BaseModel):
    labels: list[Label]


# --------------------------------------------------------------------------- #
# Job plan (deterministic)
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def load_seed_pool() -> tuple[pd.DataFrame, set[str]]:
    """Real messages from the train/val split, plus normalized test texts (for leakage checks)."""
    import train_intent_lstm as intent_lstm  # sets Keras backend; reuses the exact intent split

    df, _, y, _ = intent_lstm.prepare_data()
    idx_train, idx_val, idx_test = intent_lstm.split_indices(y)
    seeds = df.iloc[np.concatenate([idx_train, idx_val])][["id", "text", "intent_label"]].reset_index(drop=True)
    test_texts = set(df.iloc[idx_test]["text"].map(normalize))
    # A few train/val texts equal a test text up to case/punctuation; never seed from those.
    overlap = seeds["text"].map(normalize).isin(test_texts)
    if overlap.any():
        log.info("Excluding %d seeds that match a test message after normalization", overlap.sum())
    return seeds[~overlap].reset_index(drop=True), test_texts


def plan_jobs(seeds: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Allocate ~target/3 jobs across intents (sqrt-balanced so rare intents get more), then seeds."""
    rng = np.random.default_rng(seed)
    n_jobs = math.ceil(target / len(SENTIMENTS))
    counts = seeds["intent_label"].value_counts().sort_index()
    capacity = (counts * MAX_JOBS_PER_SEED).astype(int)
    weights = np.sqrt(counts)
    alloc = (weights / weights.sum() * n_jobs).round().astype(int).clip(upper=capacity)
    # Redistribute rounding / capping shortfall to intents with spare capacity, largest first.
    while alloc.sum() < n_jobs:
        spare = (capacity - alloc)[capacity > alloc]
        alloc[spare.index[np.argmax(weights[spare.index])]] += 1
    while alloc.sum() > n_jobs:
        alloc[alloc.idxmax()] -= 1

    jobs = []
    for intent, k in alloc.items():
        if k == 0:
            continue
        pool = seeds[seeds["intent_label"] == intent]
        # Cycle through shuffled seeds so each is reused as evenly as possible.
        order = np.concatenate([rng.permutation(len(pool)) for _ in range(math.ceil(k / len(pool)))])[:k]
        for i in order:
            row = pool.iloc[i]
            jobs.append({
                "seed_id": row["id"], "seed_text": row["text"], "seed_intent": intent,
                **{dim: opts[rng.integers(len(opts))] for dim, opts in STYLES.items()},
            })
    jobs = pd.DataFrame(jobs).sample(frac=1, random_state=seed).reset_index(drop=True)
    jobs.insert(0, "job_id", [f"{PROMPT_VERSION}-{seed}-{i:05d}" for i in range(len(jobs))])
    return jobs


# --------------------------------------------------------------------------- #
# API calls
# --------------------------------------------------------------------------- #
class RefusalError(RuntimeError):
    pass


def _parse(client, system: str, user: str, schema: type[BaseModel], max_tokens: int):
    response = client.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    if response.stop_reason == "refusal":
        raise RefusalError(getattr(response.stop_details, "category", None))
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise RuntimeError(f"incomplete output (stop_reason={response.stop_reason})")
    usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
    return response.parsed_output, usage, response._request_id


def run_job(client, job: dict) -> dict:
    gen_prompt = (
        f"Original message: {job['seed_text']}\n"
        f"Intent: {job['seed_intent']} ({INTENT_DESCRIPTIONS[job['seed_intent']]})\n"
        f"Style: length = {job['length']}; register = {job['register']}; quirk = {job['quirk']}\n"
        "Write the negative, neutral and positive versions."
    )
    gen, gen_usage, gen_req = _parse(client, GENERATE_SYSTEM, gen_prompt, Generation, 2000)
    variants = {v.target_sentiment: v.text.strip() for v in gen.variants}
    missing = set(SENTIMENTS) - set(variants)
    if missing:
        raise RuntimeError(f"generation missing sentiments {sorted(missing)}")

    # Blind labelling: shuffle order per job so position doesn't reveal the target tone.
    order = list(np.random.default_rng(zlib.crc32(job["job_id"].encode())).permutation(SENTIMENTS))
    texts = [variants[s] for s in order]
    label_prompt = "Label these messages:\n" + "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    labels, lab_usage, lab_req = _parse(client, LABEL_SYSTEM, label_prompt, LabelBatch, 2000)
    by_index = {l.index: l for l in labels.labels}
    if set(by_index) != set(range(len(texts))):
        raise RuntimeError("labels do not cover every message")

    return {
        **job, "model": MODEL, "prompt_version": PROMPT_VERSION,
        "request_ids": [gen_req, lab_req],
        "usage": {k: gen_usage[k] + lab_usage[k] for k in gen_usage},
        "messages": [
            {"target_sentiment": s, "text": t, **by_index[i].model_dump(exclude={"index"})}
            for i, (s, t) in enumerate(zip(order, texts))
        ],
    }


def load_cache(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["job_id"]] = rec
    return done


def run_jobs(jobs: pd.DataFrame, workers: int) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    done = load_cache(CACHE_JSONL)
    todo = [j for j in jobs.to_dict("records") if j["job_id"] not in done]
    log.info("Jobs: %d planned, %d cached, %d to run (%d workers)", len(jobs), len(done), len(todo), workers)
    if not todo:
        return

    CACHE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    tokens = {"input": 0, "output": 0}
    failures: dict[str, int] = {}
    with open(CACHE_JSONL, "a") as cache, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(run_job, client, job): job for job in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                rec = fut.result()
            except (RefusalError, RuntimeError) as e:
                failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
                log.warning("Job %s failed: %s: %s", job["job_id"], type(e).__name__, e)
                continue
            except anthropic.APIStatusError as e:
                failures[f"HTTP {e.status_code}"] = failures.get(f"HTTP {e.status_code}", 0) + 1
                log.warning("Job %s API error %s: %s", job["job_id"], e.status_code, e.message)
                if e.status_code in (401, 403):
                    raise
                continue
            with lock:
                cache.write(json.dumps(rec) + "\n")
                cache.flush()
                for k in tokens:
                    tokens[k] += rec["usage"][k]
            if n % 100 == 0 or n == len(todo):
                log.info("%d/%d jobs | tokens in=%d out=%d | cost so far $%.2f | failures %s",
                         n, len(todo), tokens["input"], tokens["output"], cost(tokens), failures or "none")


def cost(tokens: dict) -> float:
    return sum(tokens[k] / 1e6 * PRICE_PER_MTOK[k] for k in tokens)


# --------------------------------------------------------------------------- #
# Flatten cache -> labelled CSV for review / training
# --------------------------------------------------------------------------- #
def build_dataset(jobs: pd.DataFrame, test_texts: set[str]) -> pd.DataFrame:
    done = load_cache(CACHE_JSONL)
    planned = set(jobs["job_id"])
    rows = []
    for job_id, rec in done.items():
        if job_id not in planned:
            continue
        for m in rec["messages"]:
            rows.append({
                "job_id": job_id, "seed_id": rec["seed_id"], "seed_intent": rec["seed_intent"],
                "seed_text": rec["seed_text"], "style": f"{rec['length']} | {rec['register']} | {rec['quirk']}",
                "target_sentiment": m["target_sentiment"], "text": m["text"],
                "llm_intent": m["intent"], "llm_sentiment": m["sentiment"],
                "llm_is_emergency": m["is_emergency"], "llm_rationale": m["rationale"],
                "model": rec["model"], "prompt_version": rec["prompt_version"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop leaks and duplicates: anything equal to a real test message, a seed, or another synthetic text.
    norm = df["text"].map(normalize)
    seed_norm = set(df["seed_text"].map(normalize))
    leak, copy_of_seed, dup = norm.isin(test_texts), norm.isin(seed_norm), norm.duplicated()
    log.info("Dropping %d test-text matches, %d verbatim seed copies, %d duplicates",
             leak.sum(), (copy_of_seed & ~leak).sum(), (dup & ~leak & ~copy_of_seed).sum())
    df = df[~(leak | copy_of_seed | dup)].reset_index(drop=True)

    # Action from the policy on the blind labels; emergency flag escalates.
    df["action"] = [policy.table_action(i, s) for i, s in zip(df["llm_intent"], df["llm_sentiment"])]
    escalate = df["llm_is_emergency"] & (df["action"] != policy.ROUTE_EMERGENCY)
    df.loc[escalate, "action"] = policy.ROUTE_EMERGENCY
    df["policy_exception"] = escalate
    df["group"] = df["llm_intent"].map(policy.INTENT_GROUP)

    reasons = pd.DataFrame({
        "sentiment_disagrees": df["target_sentiment"] != df["llm_sentiment"],
        "intent_changed": df["seed_intent"] != df["llm_intent"],
        "emergency": df["llm_is_emergency"] | (df["llm_intent"] == "emergency"),
    })
    df["review_reasons"] = reasons.apply(lambda r: ";".join(c for c in reasons.columns if r[c]), axis=1)
    df["review_status"] = np.where(df["review_reasons"] != "", "needs_review", "pending")
    for col in ("reviewer_intent", "reviewer_sentiment", "reviewer_action", "reviewer_notes"):
        df[col] = ""
    df.insert(0, "synthetic_id", [f"S{i:05d}" for i in range(len(df))])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=TARGET_MESSAGES, help="Messages to generate (3 per job)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the job plan")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N jobs (pilot)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan and cost estimate only")
    parser.add_argument("--build-only", action="store_true", help="Rebuild the CSV from the cache only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("healthcare_ingestion", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    seeds, test_texts = load_seed_pool()
    jobs = plan_jobs(seeds, args.target, args.seed)
    run_set = jobs.head(args.limit) if args.limit else jobs
    log.info("Seed pool: %d real train/val messages | plan: %d jobs -> ~%d messages",
             len(seeds), len(jobs), 3 * len(jobs))
    log.info("Planned jobs per intent:\n%s", jobs["seed_intent"].value_counts().to_string())

    if args.dry_run:
        # ~1.0K input + ~0.4K output tokens per generate call, ~1.2K + ~0.4K per label call.
        est = cost({"input": 2200 * len(run_set), "output": 800 * len(run_set)})
        log.info("Dry run: %d jobs, estimated cost ~$%.2f with %s", len(run_set), est, MODEL)
        return
    if not args.build_only:
        run_jobs(run_set, args.workers)

    df = build_dataset(run_set, test_texts)
    if df.empty:
        log.warning("No generations cached yet.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info("Synthetic messages: %d | needs_review: %d (%.0f%%)", len(df),
             (df["review_status"] == "needs_review").sum(), 100 * (df["review_status"] == "needs_review").mean())
    log.info("Label agreement with target sentiment: %.1f%% | intent kept: %.1f%%",
             100 * (df["target_sentiment"] == df["llm_sentiment"]).mean(),
             100 * (df["seed_intent"] == df["llm_intent"]).mean())
    log.info("Actions:\n%s", df["action"].value_counts().to_string())
    log.info("Wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
