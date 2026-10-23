# ============================================================
# TWO-IMAGE DELIVERY DIAGNOSTIC (v2 files) - self-contained, run this one cell
# Test A: two solid-colour images (red, blue / blue, red)  -> are BOTH images delivered, in the right order?
# Test B: real 4 h pairs, free-text description of the shadows
# Test C: "are these identical?" on real 4 h pairs vs the same image twice
# ============================================================
import json, re, shlex, subprocess, sys
from pathlib import Path
from PIL import Image

WORK = Path("/kaggle/working")
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LABEL = "qwen3b_diag"
EXTRA = ""                                    # e.g. "--load-4bit" if you change to a 7B model

# ---- your paths first; if they moved, search /kaggle/input automatically ----
CODE = Path("/kaggle/input/datasets/mshrooom/py-files")
PROJECT = CODE / "4D_spatiotemporal" / "4D_spatiotemporal"
PAIRS = PROJECT / "pairs_v2.jsonl"
if not PAIRS.exists():
    hits = sorted(Path("/kaggle/input").rglob("pairs_v2.jsonl"))
    assert hits, "pairs_v2.jsonl not found under /kaggle/input - attach the dataset that contains it"
    PAIRS = hits[0]
RUNNER = CODE / "run_vlm_local.py"
if not RUNNER.exists():
    hits = sorted(Path("/kaggle/input").rglob("run_vlm_local.py"))
    assert hits, "run_vlm_local.py not found under /kaggle/input"
    RUNNER = hits[0]
DATA = None
for md in sorted(Path("/kaggle/input").rglob("metadata.jsonl")):
    if (md.parent / "images").exists():
        DATA = md.parent
        break
assert DATA is not None, "no folder containing both metadata.jsonl and images/ found under /kaggle/input"
print("runner:", RUNNER); print("pairs :", PAIRS); print("data  :", DATA)

# ---- build the diagnostic items (absolute image paths are fine: the runner joins them to --data) ----
WORK.mkdir(exist_ok=True)
Image.new("RGB", (448, 448), (220, 30, 30)).save(WORK / "diag_red.png")
Image.new("RGB", (448, 448), (30, 30, 220)).save(WORK / "diag_blue.png")
red, blue = str(WORK / "diag_red.png"), str(WORK / "diag_blue.png")

real = [json.loads(l) for l in open(PAIRS) if l.strip()]
real4 = [p for p in real if float(p.get("dt_bucket", -1)) == 4.0][:3] or real[:3]

COLOR_Q = ("You are given two images, Image A and Image B. What is the dominant colour of Image A and of Image B? "
           "Respond with JSON only, with the keys \"image_a_color\" and \"image_b_color\".")
DESC_Q = ("You are given two overhead images, Image A and Image B. State how many images you received. Then describe "
          "Image A and Image B separately, one sentence each, focusing on the direction and apparent length of the "
          "building shadows. Do not infer the capture time.")
SAME_Q = ("You are given two overhead images, Image A and Image B. Are Image A and Image B exactly the same image? "
          "Answer yes or no, then give one short reason.")

items = [dict(id="colorAB", image_a=red, image_b=blue, prompts={"R0": COLOR_Q}),
         dict(id="colorBA", image_a=blue, image_b=red, prompts={"R0": COLOR_Q})]
for i, p in enumerate(real4):
    items.append(dict(id=f"desc{i}", image_a=p["image_a"], image_b=p["image_b"], prompts={"R0": DESC_Q}))
    items.append(dict(id=f"diff{i}", image_a=p["image_a"], image_b=p["image_b"], prompts={"R0": SAME_Q}))
    items.append(dict(id=f"same{i}", image_a=p["image_a"], image_b=p["image_a"], prompts={"R0": SAME_Q}))
diag_pairs, diag_out = WORK / "diag2_pairs.jsonl", WORK / "diag2_out.jsonl"
with open(diag_pairs, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
if diag_out.exists():
    diag_out.unlink()

cmd = [sys.executable, str(RUNNER), "--data", str(DATA), "--pairs", str(diag_pairs), "--regimes", "R0",
       "--model", MODEL, "--label", LABEL, "--out", str(diag_out), "--max-new-tokens", "200"] + shlex.split(EXTRA)
print("\nrunning", len(items), "requests (model load + a minute or two) ...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode:
    print("RUNNER FAILED:\n", res.stderr[-3000:])
assert diag_out.exists(), "no output produced - see the runner error above"
ans = {}
for l in open(diag_out):
    r = json.loads(l)
    ans[r["id"]] = r["response"]

# ---- report ----
def colors(s):
    m = re.findall(r"\{[^{}]*\}", s)
    try:
        o = json.loads(m[0]) if m else {}
    except Exception:
        o = {}
    return str(o.get("image_a_color", "?")).lower(), str(o.get("image_b_color", "?")).lower()

print("\n" + "=" * 70 + "\nTEST A - colour delivery (expect A=red,B=blue then A=blue,B=red)\n" + "=" * 70)
for k, want in (("colorAB", ("red", "blue")), ("colorBA", ("blue", "red"))):
    got = colors(ans.get(k, ""))
    ok = want[0] in got[0] and want[1] in got[1]
    print(f"{k}: expected {want}, got {got} -> {'PASS' if ok else 'FAIL'}   raw: {ans.get(k, '')[:120]!r}")

print("\n" + "=" * 70 + "\nTEST B - descriptions of real 4 h pairs (do A and B get described differently?)\n" + "=" * 70)
for i in range(len(real4)):
    print(f"\n[pair {i}] {ans.get(f'desc{i}', '<missing>')}")

print("\n" + "=" * 70 + "\nTEST C - 'identical?' (different 4 h pair should differ; same image twice should be identical)\n" + "=" * 70)
for i in range(len(real4)):
    print(f"[pair {i}] 4 h apart : {ans.get(f'diff{i}', '<missing>')[:140]!r}")
    print(f"[pair {i}] same twice: {ans.get(f'same{i}', '<missing>')[:140]!r}")
