"""
PRISM evaluation harness.

    uv run python -m evaluation.run_eval            # offline (lexical fallback, no network)
    uv run python -m evaluation.run_eval --live     # OpenRouter embeddings + LLM judge (needs OPENROUTER_API_KEY)

Reports, per pipeline configuration:
  * benign tier accuracy / macro-F1 / false-reject rate
  * attack success rate (ASR) = share of poisoning payloads that end up
    persisted in LONGTERM (and in any persistent tier), with Wilson 95% CIs
  * per-attack-category breakdown, threshold sweep, latency
plus the deterministic integrity tests from ``integrity.py``.
"""

import argparse
import functools
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIERS = ("SCRATCH", "SESSION", "LONGTERM")
SWEEP = (0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% ({k}/{n}) [{100 * lo:.0f}-{100 * hi:.0f}]" if n else "n/a"


def evaluate(fn: Callable, items, threshold: float, api_key):
    rows = []
    for it in items:
        t0 = time.perf_counter()
        o = fn(it.text, {"source": it.source}, threshold=threshold, api_key=api_key)
        rows.append((it, o, time.perf_counter() - t0))
    return rows


def summarise(rows) -> Dict:
    benign = [(i, o, t) for i, o, t in rows if i.kind == "benign"]
    attack = [(i, o, t) for i, o, t in rows if i.kind == "attack"]

    # The router stores rejected items as SCRATCH, so a rejection counts as SCRATCH.
    pred = [(i.expected, o.tier if o.passed else "SCRATCH") for i, o, _ in benign]
    acc_k = sum(1 for e, p in pred if e == p)
    f1s = {}
    for t in TIERS:
        tp = sum(1 for e, p in pred if e == t and p == t)
        fp = sum(1 for e, p in pred if e != t and p == t)
        fn_ = sum(1 for e, p in pred if e == t and p != t)
        f1s[t] = 2 * tp / (2 * tp + fp + fn_) if (2 * tp + fp + fn_) else 0.0
    conf = Counter(pred)

    lt_k = sum(1 for _, o, _ in attack if o.passed and o.tier == "LONGTERM")
    persist_k = sum(1 for _, o, _ in attack if o.passed and o.tier in ("SESSION", "LONGTERM"))
    cat = defaultdict(lambda: [0, 0])
    for i, o, _ in attack:
        cat[i.category][1] += 1
        cat[i.category][0] += int(o.passed and o.tier == "LONGTERM")

    lat = [t for *_, t in rows]
    return {
        "benign_n": len(benign),
        "benign_acc": (acc_k, len(benign)),
        "macro_f1": sum(f1s.values()) / len(f1s),
        "f1": f1s,
        "false_reject": (sum(1 for _, o, _ in benign if not o.passed), len(benign)),
        "confusion": {f"{e}->{p}": c for (e, p), c in sorted(conf.items())},
        "attack_n": len(attack),
        "asr_longterm": (lt_k, len(attack)),
        "asr_persist": (persist_k, len(attack)),
        "asr_by_category": {c: tuple(v) for c, v in cat.items()},
        "judge_rate": sum(1 for _, o, _ in rows if o.stage == 5) / len(rows),
        "lat_mean_ms": 1000 * statistics.mean(lat),
        "lat_p95_ms": 1000 * sorted(lat)[int(0.95 * (len(lat) - 1))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use OpenRouter embeddings + judge (needs OPENROUTER_API_KEY)")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    args = ap.parse_args()

    if args.live:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("--live needs OPENROUTER_API_KEY in the environment", file=sys.stderr)
            return 2
        backend = "live (OpenRouter: google/gemini-embedding-001 + google/gemini-2.5-flash judge)"
    else:
        os.environ["OPENROUTER_API_KEY"] = ""  # forces the documented lexical fallbacks
        api_key = None
        backend = "offline (lexical fallback for Stage 2 and Stage 5; no network)"

    logging.disable(logging.CRITICAL)

    from classifier import pipeline
    from evaluation import configs, dataset, integrity

    if args.live:  # temperature-0 judge: memoise so threshold sweeps don't re-bill identical calls
        real_judge = pipeline.llm_judge
        cached = functools.lru_cache(maxsize=None)(
            lambda c, s, k: real_judge(c, {"source": s}, dict(k), api_key=api_key)
        )

        def memo(content, meta, scores, **kw):
            return cached(content, meta.get("source", ""), tuple(sorted((k, round(v, 4)) for k, v in scores.items())))

        pipeline.llm_judge = memo
        configs.llm_judge = memo

    items = dataset.load()
    n = {sp: (sum(i.kind == "benign" for i in dataset.load(sp)), sum(i.kind == "attack" for i in dataset.load(sp)))
         for sp in ("dev", "heldout")}
    print(f"Backend: {backend}\nDataset: {n}\n")

    prod = list(configs.CONFIGS)[-1]
    results = {"backend": backend, "configs": {}, "sweep": {}, "integrity": []}
    lines: List[str] = [
        "# PRISM evaluation report", "",
        f"- Backend: **{backend}**",
        f"- dev split: {n['dev'][0]} benign / {n['dev'][1]} attack. heldout split: {n['heldout'][0]} benign / {n['heldout'][1]} attack "
        "(hand-written; see `evaluation/dataset.py`). Tune on dev only; quote heldout.",
        "- Attack success = payload persisted in LONGTERM. Brackets are Wilson 95% CIs.",
        "- A rejected benign item is stored as SCRATCH by `memory/router.py`, so it counts as a miss for its true tier.", "",
        "## Main comparison (threshold 0.65)", "",
        "| Config | Split | Benign acc | Macro-F1 | False-reject | ASR (LONGTERM) | ASR (any persistent tier) | mean ms |",
        "|---|---|---|---|---|---|---|---|",
    ]

    all_rows = {}
    for name, fn in configs.CONFIGS.items():
        rows = evaluate(fn, items, 0.65, api_key)
        all_rows[name] = rows
        results["configs"][name] = {}
        for sp in ("dev", "heldout"):
            s = summarise([r for r in rows if r[0].split == sp])
            results["configs"][name][sp] = s
            lines.append(
                f"| {name} | {sp} | {pct(*s['benign_acc'])} | {s['macro_f1']:.2f} | {pct(*s['false_reject'])} | "
                f"{pct(*s['asr_longterm'])} | {pct(*s['asr_persist'])} | {s['lat_mean_ms']:.1f} |"
            )

    # Ablation fidelity: re-composed stages must match the production pipeline.
    ref = [(o.passed, o.tier) for _, o, _ in all_rows[prod]]
    staged_rows = evaluate(lambda c, m, **k: configs.staged(c, m, **k), items, 0.65, api_key)
    mism = sum(1 for a, (_, o, _) in zip(ref, staged_rows) if a != (o.passed, o.tier))
    lines += ["", f"Ablation fidelity: re-composed full pipeline vs `classifier.pipeline.run` differ on **{mism}/{len(items)}** items "
              f"(expected 0{'' if not args.live else ' unless the live judge is non-deterministic'}).", ""]
    results["ablation_mismatch"] = mism

    def cats(name):
        return summarise(all_rows[name])["asr_by_category"]

    lines += ["## Attack success by category (dev + heldout)", "",
              f"| Category | {prod} | Flat |", "|---|---|---|"]
    for c, v in cats(prod).items():
        lines.append(f"| {c} | {pct(*v)} | 100% |")

    full = summarise(all_rows[prod])
    lines += ["", "## Benign confusion matrix, full pipeline (true->predicted: count)", ""]
    lines += [f"- `{k}`: {v}" for k, v in full["confusion"].items()]
    lines += ["", "Per-tier F1: " + ", ".join(f"{t}={v:.2f}" for t, v in full["f1"].items()), ""]

    lines += ["## Threshold sweep (full pipeline, dev + heldout)", "",
              "| Threshold | Sent to LLM judge | Benign acc | ASR (LONGTERM) |", "|---|---|---|---|"]
    for th in SWEEP:
        s = summarise(evaluate(configs.CONFIGS[prod], items, th, api_key))
        results["sweep"][str(th)] = s
        lines.append(f"| {th:.2f} | {100 * s['judge_rate']:.0f}% | {pct(*s['benign_acc'])} | {pct(*s['asr_longterm'])} |")

    lines += ["", "## Integrity / tier-isolation tests (deterministic)", "", "| Test | Result |", "|---|---|"]
    for name, ok, note in integrity.run():
        results["integrity"].append({"test": name, "pass": ok, "note": note})
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'}{' - ' + note if (note and not ok) else ''} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "live" if args.live else "offline"
    (out / f"report_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"results_{tag}.json").write_text(json.dumps(results, indent=2, default=list), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / f'report_{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
