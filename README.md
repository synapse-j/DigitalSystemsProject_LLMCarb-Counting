# AI Carbohydrate Counting for Type 1 Diabetes
**Jake Richardson-Price — Dissertation Project**

Compares a cascaded ResNet-50 ML pipeline against large language models
(Claude, GPT-4o) for classifying food images into carbohydrate ranges,
to support insulin dosing in individuals with type 1 diabetes.

---

## Project Structure

```
carb-counting-t1d/
├── config.py                  ← all settings (edit this first)
├── run.py                     ← single entry point
├── requirements.txt
├── .env.example               ← copy to .env and add your API keys
│
├── pipeline/
│   ├── dataset.py             ← download Food-101, build datasets
│   ├── train_stage1.py        ← ResNet-50 food classifier
│   ├── train_stage2.py        ← ResNet-50 portion estimator
│   └── cascade.py             ← combine both → carb range prediction
│
├── evaluation/
│   ├── llm_eval.py            ← send images to Claude / GPT-4o
│   └── compare.py             ← dissertation figures + summary table
│
├── data/                      ← created automatically (not on GitHub)
├── models/                    ← saved checkpoints (not on GitHub)
└── results/                   ← output figures and JSON (not on GitHub)
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/carb-counting-t1d.git
cd carb-counting-t1d

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
copy .env.example .env       # Windows
cp .env.example .env         # Mac / Linux
# Then open .env and fill in your keys
```

---

## Running the Project

```bash
# Run everything from start to finish
python run.py

# Or run individual stages
python run.py --stage data       # download + prepare dataset (~30 mins)
python run.py --stage train      # train ML models (~4-8 hrs on CPU)
python run.py --stage cascade    # ML pipeline evaluation
python run.py --stage llm --limit 25   # cheap LLM test (~£0.50)
python run.py --stage llm        # full LLM evaluation
python run.py --stage compare    # generate dissertation figures
```

---

## Carbohydrate Ranges

| Range | Grams  | Examples                              |
|-------|--------|---------------------------------------|
| 0     | 0–20g  | Salads, eggs, grilled proteins        |
| 1     | 21–40g | Soups, light sandwiches               |
| 2     | 41–60g | Standard pasta, pizza slice, rice     |
| 3     | 61–80g | Large burger, cake slice, big pasta   |
| 4     | 81g+   | Large desserts, multiple portions     |

Clinical threshold: ±1 range (≈ ±20g) is considered acceptable per Özkaya et al. (2026).

---

## API Keys

Get keys from:
- Anthropic Claude: https://console.anthropic.com
- OpenAI GPT-4o: https://platform.openai.com

Estimated cost for full LLM evaluation:
| Images | Claude  | GPT-4o  |
|--------|---------|---------|
| 25     | ~£0.50  | ~£0.75  |
| 100    | ~£2.00  | ~£3.00  |

Always test with `--limit 25` first.

---

## References

- Özkaya et al. (2026). Carbohydrate counting in traditional Turkish fast foods for individuals with type 1 diabetes.
- Alfonsi et al. (2020). Carbohydrate counting app using image recognition for youth with T1D.
- O'Hara et al. (2025). An evaluation of ChatGPT for nutrient content estimation.
- Bossard et al. (2014). Food-101. ECCV.
