"""
evaluation/llm_eval.py
=======================
Sends food images to LLMs and collects carb range predictions.

Supports:
  - Anthropic Claude  (--provider anthropic)
  - OpenAI GPT-4o     (--provider openai)
  - Both              (--provider both)

Both use the IDENTICAL prompt so results are directly comparable.

API keys go in a .env file in the project root (never committed to Git):
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, RESULTS_DIR, CARB_RANGE_LABELS

RANGE_LABELS    = list(CARB_RANGE_LABELS.values())
MAX_RETRIES     = 3
RETRY_DELAY     = 5

# ── Identical prompt sent to ALL LLMs ─────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical dietitian specialising in type 1 diabetes.

Estimate the total carbohydrate content of the meal in the image,
considering BOTH what the food is AND the portion size visible.

Respond ONLY with valid JSON — no text outside the JSON object:

{
  "food_identified": "<food name>",
  "portion_assessment": "small" | "medium" | "large",
  "estimated_carbs_grams": <integer>,
  "carb_range": <integer 0-4>,
  "confidence": "low" | "medium" | "high",
  "reasoning": "<one sentence>"
}

Carb range values:
  0 = 0 to 20 grams    (salads, eggs, plain proteins)
  1 = 21 to 40 grams   (soups, light sandwiches)
  2 = 41 to 60 grams   (standard pasta, pizza slice, rice)
  3 = 61 to 80 grams   (large burger, cake slice, big pasta)
  4 = 81 grams or more (large desserts, multiple portions)"""


def encode_image(path: str) -> tuple:
    ext  = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def grams_to_range(g: int) -> int:
    if g <= 20:   return 0
    elif g <= 40: return 1
    elif g <= 60: return 2
    elif g <= 80: return 3
    else:         return 4


def parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        cr = int(parsed.get("carb_range", -1))
        if cr not in range(5):
            cr = grams_to_range(int(parsed.get("estimated_carbs_grams", 50)))
        parsed["carb_range"] = cr
        return parsed
    except (json.JSONDecodeError, ValueError):
        nums = re.findall(r'\b([0-4])\b', text)
        return {"carb_range": int(nums[0]) if nums else 2, "parse_error": True}


# ──────────────────────────────────────────────
# PROVIDER FUNCTIONS
# ──────────────────────────────────────────────

def query_anthropic(client, path: str) -> dict:
    b64, mime = encode_image(path)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": "Estimate the carbohydrate range."},
                ]}],
            )
            return parse_response(resp.content[0].text)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"carb_range": -1, "error": str(e)}


def query_openai(client, path: str) -> dict:
    b64, mime = encode_image(path)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                        {"type": "text", "text": "Estimate the carbohydrate range."},
                    ]},
                ],
                max_tokens=300,
                temperature=0,
            )
            return parse_response(resp.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"carb_range": -1, "error": str(e)}


# ──────────────────────────────────────────────
# EVALUATION RUNNER
# ──────────────────────────────────────────────

def collect_images(limit: int = None) -> list:
    eval_dir = Path(DATASET_DIR) / "final_eval"
    samples  = []
    for range_dir in sorted(eval_dir.iterdir()):
        if not range_dir.is_dir():
            continue
        label = int(range_dir.name.split("_")[1])
        for img in range_dir.glob("*.jpg"):
            samples.append((str(img), label))

    if limit:
        per_class = max(1, limit // 5)
        buckets   = {i: [] for i in range(5)}
        for path, label in samples:
            buckets[label].append((path, label))
        balanced = []
        for i in range(5):
            balanced.extend(buckets[i][:per_class])
        return balanced
    return samples


def run_evaluation(provider: str, query_fn, model_name: str,
                   samples: list, save_tag: str):

    print(f"\n{'═'*60}")
    print(f"  LLM: {model_name}")
    print(f"  Images: {len(samples)}  |  Est. cost: £{len(samples)*0.02:.2f}")
    print(f"{'═'*60}")

    results     = []
    true_labels = []
    pred_labels = []
    errors      = 0

    for i, (img_path, true_label) in enumerate(samples):
        response   = query_fn(img_path)
        pred_range = response.get("carb_range", -1)

        if pred_range == -1:
            errors += 1
        else:
            true_labels.append(true_label)
            pred_labels.append(pred_range)

        results.append({
            "image":         img_path,
            "true_range":    true_label,
            "pred_range":    pred_range,
            "food_id":       response.get("food_identified", ""),
            "portion":       response.get("portion_assessment", ""),
            "carbs_g":       response.get("estimated_carbs_grams"),
            "confidence":    response.get("confidence", ""),
            "reasoning":     response.get("reasoning", ""),
            "error":         "error" in response,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            acc = (sum(t == p for t, p in zip(true_labels, pred_labels))
                   / len(true_labels)) if true_labels else 0
            print(f"  [{i+1:4d}/{len(samples)}]  acc={acc:.3f}  "
                  f"true={true_label}  pred={pred_range}  errors={errors}")

        time.sleep(0.8)

    labels = np.array(true_labels)
    preds  = np.array(pred_labels)
    acc      = (labels == preds).mean()
    clin_acc = (np.abs(labels - preds) <= 1).mean()
    report   = classification_report(labels, preds, target_names=RANGE_LABELS,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Accuracy:                 {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Clinically acceptable:    {clin_acc:.3f}  ({clin_acc*100:.1f}%)")
    print(f"  API errors:               {errors}")

    # Confusion matrix
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(RANGE_LABELS, rotation=30, ha="right")
    ax.set_yticklabels(RANGE_LABELS)
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"cm_{save_tag}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix → {cm_path}")

    output = {
        "model":                                   model_name,
        "test_accuracy":                           round(float(acc), 4),
        "clinically_acceptable_accuracy_±1_range": round(float(clin_acc), 4),
        "classification_report":                   report,
        "confusion_matrix":                        cm.tolist(),
        "per_image_results":                       results,
    }
    out_path = os.path.join(RESULTS_DIR, f"{save_tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Results → {out_path}")
    return output


def evaluate(provider: str = "anthropic", limit: int = None):
    samples = collect_images(limit=limit)
    print(f"\nTest images: {len(samples)}")

    if provider in ("anthropic", "both"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("\nERROR: ANTHROPIC_API_KEY not set in .env")
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                run_evaluation("anthropic",
                               lambda p: query_anthropic(client, p),
                               "Claude (Anthropic)", samples, "claude")
            except ImportError:
                print("Run: pip install anthropic")

    if provider in ("openai", "both"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("\nERROR: OPENAI_API_KEY not set in .env")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key)
                run_evaluation("openai",
                               lambda p: query_openai(client, p),
                               "GPT-4o (OpenAI)", samples, "gpt4o")
            except ImportError:
                print("Run: pip install openai")
