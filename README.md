# AI Carbohydrate Counting for Type 1 Diabetes
**Jake Richardson-Price — BSc Digital Systems — Dissertation Project**

---

## Overview

Type 1 diabetes (T1D) affects over 225,000 people in the UK. Individuals with T1D must manually estimate the carbohydrate content of every meal to calculate the correct insulin dose — a cognitively demanding task that directly impacts blood glucose control and quality of life.

This project investigates whether artificial intelligence can reliably estimate carbohydrate content from food images, and compares two fundamentally different AI approaches:

- **A cascaded machine learning pipeline** — two fine-tuned ResNet-50 convolutional neural networks working in sequence: one identifies the food, a second estimates the portion size, and a lookup table combines both into a carbohydrate range
- **Large language models** — Claude (Anthropic) and GPT-4o (OpenAI) asked to estimate carbohydrate content directly from the same images in a single step

Performance is evaluated using classification accuracy, per-class F1 score, and a **clinically acceptable accuracy metric** (proportion of predictions within ±1 carbohydrate range, approximately ±20g), based on the clinical threshold established by Özkaya et al. (2026).

---

## Research Question

*Can a purpose-built cascaded ML pipeline match or exceed the carbohydrate estimation accuracy of general-purpose large language models when classifying food images into clinically relevant carbohydrate ranges?*

---

## Carbohydrate Ranges

Predictions are classified into five ranges reflecting clinical dosing thresholds:

| Range | Grams   | Clinical Context                          |
|-------|---------|-------------------------------------------|
| 0     | 0–20g   | Very low — salads, eggs, plain proteins   |
| 1     | 21–40g  | Low — soups, light sandwiches             |
| 2     | 41–60g  | Medium — pasta, pizza slice, rice dish    |
| 3     | 61–80g  | High — large burger, cake, big pasta      |
| 4     | 81g+    | Very high — large desserts, big portions  |

An error of ±1 range (≈±20g) is considered clinically acceptable. Errors beyond ±20g can meaningfully affect blood glucose levels (Özkaya et al., 2026).

---

## System Architecture

```
Food Image
    │
    ├──► Stage 1: ResNet-50 (Food Classifier)
    │         Identifies what food is in the image
    │         101 classes from the Food-101 dataset
    │
    ├──► Stage 2: ResNet-50 (Portion Estimator)
    │         Estimates how much food is present
    │         3 classes: small / medium / large
    │
    └──► Lookup Table
              Combines food category + portion size
              → Carbohydrate range prediction (0–4)


Food Image ──► Claude / GPT-4o (LLM)
                    Single-step carb range prediction
                    No specialised training required
```

The cascade separates two visually distinct tasks — food identity and portion size — that compete for the same feature space in a single model. This is the core methodological hypothesis tested by the project.

---

## Project Structure

```
carb-counting-t1d/
├── config.py                  ← all settings and carb lookup table
├── run.py                     ← single entry point for all stages
├── requirements.txt
├── .env.example               ← copy to .env, add your API keys
│
├── pipeline/
│   ├── dataset.py             ← downloads Food-101, builds datasets
│   ├── train_stage1.py        ← trains ResNet-50 food classifier
│   ├── train_stage2.py        ← trains ResNet-50 portion estimator
│   └── cascade.py             ← runs full pipeline, generates metrics
│
├── evaluation/
│   ├── llm_eval.py            ← evaluates Claude and/or GPT-4o
│   └── compare.py             ← generates dissertation figures and table
│
├── data/                      ← created automatically (not on GitHub)
├── models/                    ← saved model checkpoints (not on GitHub)
└── results/                   ← output figures and JSON (not on GitHub)
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/carb-counting-t1d.git
cd carb-counting-t1d

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env           # Mac / Linux
copy .env.example .env         # Windows
# Open .env and fill in your Anthropic and/or OpenAI API keys
```

---

## Running the Project

All stages are controlled through a single entry point:

```bash
# Run the full pipeline from start to finish
python run.py

# Or run individual stages
python run.py --stage data       # Download Food-101 and prepare datasets
python run.py --stage train      # Train both ResNet-50 models
python run.py --stage cascade    # Run ML pipeline evaluation
python run.py --stage llm --limit 25   # Test LLM evaluation cheaply first
python run.py --stage llm        # Full LLM evaluation
python run.py --stage compare    # Generate dissertation figures
```

### Estimated runtimes

| Stage | CPU | GPU |
|-------|-----|-----|
| Data preparation | ~30 mins | ~30 mins |
| Stage 1 training | ~4–6 hrs | ~30 mins |
| Stage 2 training | ~2–3 hrs | ~20 mins |
| Cascade evaluation | ~10 mins | ~5 mins |
| LLM evaluation (100 images) | ~15 mins | ~15 mins |

A GPU is strongly recommended for training. Google Colab (free tier) provides sufficient GPU time for a full training run.

---

## Dataset

**Food-101** is downloaded automatically when you run `--stage data`. No manual download required.

- 101 food categories, 1,000 images each (101,000 images total)
- ~5 GB download from ETH Zurich servers
- Free for academic research use
- Citation: Bossard et al. (2014), ECCV

Carbohydrate reference values in `config.py` are sourced from the USDA FoodData Central database and cross-referenced with Nutritionix. Values represent a typical single serving with documented ranges reflecting recipe variation.

---

## API Keys

LLM evaluation (optional) requires API access:

| Provider | Where to get a key | Estimated cost (100 images) |
|----------|-------------------|----------------------------|
| Anthropic (Claude) | console.anthropic.com | ~£2 |
| OpenAI (GPT-4o) | platform.openai.com | ~£3 |

Always test with `--limit 25` before running the full evaluation set.

---

## Output Files

Running the full pipeline produces the following in `results/`:

```
results/
├── fig1_accuracy.png             ← main comparison bar chart
├── fig2_f1_per_class.png         ← per-class F1 scores
├── fig3_radar.png                ← multi-metric radar chart
├── cm_cascaded_resnet-50.png     ← ML pipeline confusion matrix
├── cm_claude.png                 ← Claude confusion matrix
├── cm_gpt4o.png                  ← GPT-4o confusion matrix
├── summary_table.txt             ← copy-paste results table
├── cascade_results.json
├── claude_results.json
├── gpt4o_results.json
└── all_results.json
```

---

## References

- Özkaya, V., Eren, E., Özgen Özkaya, Ş., & Özkaya, G. (2026). Carbohydrate counting in traditional Turkish fast foods for individuals with type 1 diabetes: Can artificial intelligence models replace dietitians? *Nutrition*, 142, 112986.
- Alfonsi, J. E. et al. (2020). Carbohydrate counting app using image recognition for youth with Type 1 diabetes. *JMIR MHealth and UHealth*, 8(10), e22074.
- O'Hara, C. et al. (2025). An evaluation of ChatGPT for nutrient content estimation from meal photographs. *Nutrients*, 17(4), 607.
- Bossard, L. et al. (2014). Food-101 — Mining discriminative components with random forests. *ECCV*.
