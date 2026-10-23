#!/usr/bin/env python3
"""
run_vlm.py - query a vision-language model on the pairs and write predictions for score.py.

Works with any server that speaks the OpenAI chat-completions protocol with image inputs:
  * OpenAI            --base-url https://api.openai.com/v1                     --api-key-env OPENAI_API_KEY
  * local vLLM        --base-url http://localhost:8000/v1                       --api-key-env VLLM_KEY (any string)
  * other providers that expose an OpenAI-compatible endpoint (check their docs for the URL and model name)

Install:  pip install openai

Examples
  export OPENAI_API_KEY=...            # (PowerShell: $env:OPENAI_API_KEY="...")
  python run_vlm.py --data data --pairs pairs.jsonl --model <model-name> --label gpt \
        --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --out preds_gpt.jsonl
  # quick test on 10 pairs first:  add  --max-pairs 10

The run is resumable: lines already present in --out (same id + regime) are skipped.
Temperature is 0. The raw text of every answer is stored, so parsing problems stay visible in score.py
(unparseable answers count as wrong and are reported as the invalid rate).
"""
import argparse
import base64
import json
import os
import sys
import time


def data_url(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True, help="model name exactly as the endpoint expects it")
    ap.add_argument("--label", default=None, help="name stored in the predictions (default: --model)")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--regimes", default="R0,R1,R2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=None, help="first N pairs (format test only: they all come from one scene)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random subset of about N pairs for a representative pilot (A/B-swapped twins are kept together)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"environment variable {args.api_key_env} is not set")
    client = OpenAI(base_url=args.base_url, api_key=key)
    label = args.label or args.model
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    pairs = [json.loads(l) for l in open(args.pairs)]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if args.sample:
        import random
        rng = random.Random(args.seed)
        base = [p for p in pairs if not p.get("twin_of")]
        chosen = {p["id"] for p in rng.sample(base, min(len(base), max(1, args.sample // 2)))}
        pairs = [p for p in pairs if p["id"] in chosen or p.get("twin_of") in chosen]
        print(f"[run_vlm] random pilot: {len(pairs)} pairs (seed {args.seed})", flush=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["id"], r["regime"]))

    n_new = 0
    with open(args.out, "a") as out:
        for i, p in enumerate(pairs):
            img_a = data_url(os.path.join(args.data, p["image_a"].replace("\\", "/")))
            img_b = data_url(os.path.join(args.data, p["image_b"].replace("\\", "/")))
            for regime in regimes:
                if (p["id"], regime) in done:
                    continue
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": p["prompts"][regime]},
                    {"type": "text", "text": "Image A:"}, {"type": "image_url", "image_url": {"url": img_a}},
                    {"type": "text", "text": "Image B:"}, {"type": "image_url", "image_url": {"url": img_b}},
                ]}]
                text = None
                for attempt in range(args.retries):
                    try:
                        resp = client.chat.completions.create(
                            model=args.model, messages=messages, temperature=0, max_tokens=args.max_tokens)
                        text = resp.choices[0].message.content or ""
                        break
                    except Exception as exc:  # noqa: BLE001 - rate limits, timeouts, 5xx
                        wait = 2 ** attempt
                        print(f"[run_vlm] pair {p['id']} {regime}: {type(exc).__name__}: {exc} - retry in {wait}s",
                              file=sys.stderr)
                        time.sleep(wait)
                if text is None:
                    text = ""  # recorded as an invalid answer, never silently dropped
                out.write(json.dumps(dict(id=p["id"], regime=regime, model=label, response=text)) + "\n")
                out.flush()
                n_new += 1
            if (i + 1) % 25 == 0:
                print(f"[run_vlm] {i + 1}/{len(pairs)} pairs", flush=True)
    print(f"[run_vlm] wrote {n_new} new answers to {args.out}")


if __name__ == "__main__":
    main()
