#!/usr/bin/env python3
"""
run_vlm_local.py - run an open vision-language model LOCALLY (Kaggle / any CUDA GPU), no API key.

Uses Hugging Face transformers' generic image-text-to-text interface, so it works for models whose
processor ships a chat template with image placeholders and that accept several images per prompt
(e.g. Qwen2.5-VL, LLaVA-OneVision-hf, SmolVLM, Gemma-3, InternVL-hf variants). It has NOT been run
against any real model in the environment where it was written - do the 2-pair smoke test below first.

    pip install -q transformers accelerate pillow bitsandbytes    # Kaggle: usually already present

    # 1) smoke test: 2 pairs + 4 probes, look at the printed raw answers
    python run_vlm_local.py --data data --pairs pairs.jsonl --probes probes.jsonl \
        --model Qwen/Qwen2.5-VL-3B-Instruct --label qwen25vl3b --out smoke.jsonl --max-pairs 2 --probe-max 4
    # 2) pilot: random 100 pairs (twins kept together) + all probes
    python run_vlm_local.py ... --sample 100
    # 3) full run: drop --sample (resumable: re-running skips answers already in --out)

Output lines: {"id", "regime" (R0/R1/R2 for pairs, PROBE for probes), "model", "response"}.
Memory: 7B models in fp16 need ~16 GB (device_map="auto" spreads them over 2 GPUs); add --load-4bit
for a single 16 GB GPU or if you run out of memory. T4 GPUs have no native bf16: default dtype is fp16;
if answers come out as garbage/NaN try --dtype bf16 or --load-4bit. [VERIFY on your hardware]
--backend dummy runs the whole pipeline without any model (plumbing test).
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation / fewer OOMs
import random
import sys
import time


class DummyBackend:
    def generate(self, prompt, labels, paths):
        if len(paths) in (0, 2):
            return json.dumps({"earlier": "A", "elapsed_hours": 2.0})
        return json.dumps({"shadow_direction": "N"})


def load_model_any(transformers, name, kwargs):
    """Try the generic image-text-to-text class first, then the newer multimodal and the older vision2seq classes."""
    last = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - unrecognised config, missing weights class, ...
            last = exc
            print(f"[local] {cls_name} could not load {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    raise last if last else RuntimeError("no suitable transformers Auto class found")


class HFBackend:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoProcessor
        kwargs = dict(device_map="auto", trust_remote_code=args.trust_remote_code)
        dtype = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)[args.dtype]
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = dtype
        print(f"[local] loading {args.model} ...", flush=True)
        self.model = load_model_any(transformers, args.model, kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        self.max_new_tokens = args.max_new_tokens
        self.resize = args.resize
        self.torch = torch
        print(f"[local] loaded. device map: {getattr(self.model, 'hf_device_map', 'single device')}", flush=True)

    def generate(self, prompt, labels, paths):
        from PIL import Image
        content = [{"type": "text", "text": prompt}]
        for lab in labels:  # no labels when running text-only
            content += [{"type": "text", "text": lab}, {"type": "image"}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in paths]  # empty list for --text-only
        if self.resize:  # disclose in the paper: the model sees images downscaled to resize x resize
            images = [im.resize((self.resize, self.resize), Image.BICUBIC) for im in images]
        if images:
            inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        else:  # text-only: tests the language model alone, without any image
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        try:
            inputs = inputs.to(self.model.device, self.model.dtype)
        except Exception:  # noqa: BLE001 - some processors return non-float tensors only
            inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def build_items(args):
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    items = []
    if args.pairs:
        pairs = [json.loads(l) for l in open(args.pairs)]
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        if args.sample:
            rng = random.Random(args.seed)
            base = [p for p in pairs if not p.get("twin_of")]
            chosen = {p["id"] for p in rng.sample(base, min(len(base), max(1, args.sample // 2)))}
            pairs = [p for p in pairs if p["id"] in chosen or p.get("twin_of") in chosen]
            print(f"[local] random pilot: {len(pairs)} pairs (seed {args.seed})", flush=True)
        for p in pairs:
            for regime in regimes:
                if args.text_only:
                    items.append(dict(id=p["id"], regime=regime, images=[], labels=[], prompt=p["prompts"][regime]))
                else:
                    items.append(dict(id=p["id"], regime=regime,
                                      images=[p["image_a"].replace("\\", "/"), p["image_b"].replace("\\", "/")],
                                      labels=["Image A:", "Image B:"], prompt=p["prompts"][regime]))
    if args.probes:
        probes = [json.loads(l) for l in open(args.probes)]
        if args.probe_max:
            probes = probes[:args.probe_max]
        for q in probes:
            items.append(dict(id=q["id"], regime="PROBE", images=[q["image"].replace("\\", "/")],
                              labels=["Image:"], prompt=q["prompt"]))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--model", default="dummy", help="Hugging Face model id (ignored for --backend dummy)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default="R0,R1,R2")
    ap.add_argument("--max-pairs", type=int, default=None, help="first N pairs (smoke test only)")
    ap.add_argument("--sample", type=int, default=None, help="random subset of ~N pairs, twins kept together")
    ap.add_argument("--probe-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="hf", choices=["hf", "dummy"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--text-only", action="store_true", help="do NOT send images (pair prompts only); isolates the language model")
    ap.add_argument("--resize", type=int, default=None,
                    help="downscale every image to N x N before the model (use e.g. 384 if a model runs out of memory)")
    args = ap.parse_args()
    if not (args.pairs or args.probes):
        sys.exit("give --pairs and/or --probes")

    label = args.label or args.model
    items = build_items(args)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if r.get("error") or not str(r.get("response", "")).strip():
                continue  # failed earlier -> retry now
            done.add((r["id"], r["regime"]))
    todo = [it for it in items if (it["id"], it["regime"]) not in done]
    print(f"[local] {len(items)} requests, {len(items) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    backend = DummyBackend() if args.backend == "dummy" else HFBackend(args)

    t0, n, n_err, first_err = time.time(), 0, 0, None
    with open(args.out, "a") as out:
        for it in todo:
            paths = [os.path.join(args.data, p) for p in it["images"]]
            rec = dict(id=it["id"], regime=it["regime"], model=label)
            try:
                text = backend.generate(it["prompt"], it["labels"], paths)
            except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM or an unsupported multi-image input
                err = f"{type(exc).__name__}: {exc}"[:400]
                n_err += 1
                first_err = first_err or err
                if n_err <= 3:
                    print(f"[local] ERROR on {it['id']} {it['regime']}: {err}", flush=True)
                text, rec["error"] = "", err  # recorded as an invalid answer WITH the reason, run continues
            rec["response"] = text
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            if n <= 3 or n % 25 == 0:
                print(f"[local] {n}/{len(todo)}  {(time.time() - t0) / n:.1f}s/request  last answer: {text[:100]!r}", flush=True)
    print(f"[local] done: {n} answers in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    if n_err:
        print(f"[local] ERRORS: {n_err} of {n} requests failed. First error: {first_err}")


if __name__ == "__main__":
    main()
