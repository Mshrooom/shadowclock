# ============================================================
# MINI-PILOT FOR SEVERAL MODELS (v2 files): 40 random pairs (R1 only, twins kept) + 100 probes per model
# Self-contained. A model that fails to load is reported and skipped. Same pairs for every model (seed 0).
# ============================================================
import json, shlex, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
MODELS = [  # (Hugging Face id, short label, extra runner flags)  - ids NOT verified, check each on huggingface.co
    ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen7b", ""),                # if answers are garbage/NaN on T4: "--load-4bit"
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llavaov7b", ""),
    ("HuggingFaceTB/SmolVLM-Instruct", "smolvlm", ""),
]
N_PAIRS, N_PROBES = 40, 100

def first(pattern, must_have=None):
    for p in sorted(Path("/kaggle/input").rglob(pattern)):
        if must_have is None or (p.parent / must_have).exists():
            return p
    raise FileNotFoundError(f"{pattern} not found under /kaggle/input")

RUNNER, EVAL = first("run_vlm_local.py"), first("evaluate_hypotheses.py")
PAIRS, PROBES = first("pairs_v2.jsonl"), first("probes_v2.jsonl")
DATA = first("metadata.jsonl", must_have="images").parent
print("data:", DATA, "\npairs:", PAIRS, "\nprobes:", PROBES)

finished = []
for hf_id, label, extra in MODELS:
    out = WORK / f"mini_{label}.jsonl"
    cmd = [sys.executable, str(RUNNER), "--data", str(DATA), "--pairs", str(PAIRS), "--probes", str(PROBES),
           "--regimes", "R1", "--sample", str(N_PAIRS), "--probe-max", str(N_PROBES),
           "--model", hf_id, "--label", label, "--out", str(out)] + shlex.split(extra)
    print(f"\n=== {label} ({hf_id}) ===", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    speed = [l for l in res.stdout.splitlines() if "s/request" in l][-1:]
    print("  speed line:", speed[0][:110] if speed else "none", f"| wall {(time.time() - t0) / 60:.1f} min")
    if res.returncode or not out.exists():
        print("  FAILED - last stderr lines:\n", res.stderr[-1500:])
        continue
    rows = [json.loads(l) for l in open(out)]
    empty = sum(1 for r in rows if not r["response"].strip())
    print(f"  answers: {len(rows)}, empty (runner errors such as out-of-memory): {empty}")
    print("  sample answers:", [r["response"][:70].replace("\n", " ") for r in rows[:2]])
    finished.append(out)

if finished:
    cmd = [sys.executable, str(EVAL), "--pairs", str(PAIRS), "--probes", str(PROBES), "--allow-partial"]
    for o in finished:
        cmd += ["--preds", str(o)]
    print("\n" + "=" * 70 + "\nEVALUATION (R1 only, so H1 and H3 print NOT SCORED by design)\n" + "=" * 70)
    ev = subprocess.run(cmd, capture_output=True, text=True)
    print(ev.stdout, ev.stderr[-1500:] if ev.returncode else "")
else:
    print("no model finished - nothing to evaluate")
