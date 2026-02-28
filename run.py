"""
run.py — Single entry point for the entire project
====================================================
Runs all stages in the correct order. You can run everything
at once or pick individual stages.

Usage:
    python run.py                          # runs everything
    python run.py --stage data             # download + prepare dataset only
    python run.py --stage train            # train both ML models
    python run.py --stage cascade          # run ML cascade evaluation
    python run.py --stage llm              # run LLM evaluation
    python run.py --stage compare          # generate final figures
    python run.py --stage llm --limit 25   # cheap LLM test (25 images)
    python run.py --provider openai        # use OpenAI instead of Anthropic
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file automatically
load_dotenv()


def stage_data():
    print("\n" + "="*60)
    print("  STAGE: Prepare Dataset")
    print("="*60)
    from pipeline.dataset import download_food101, build_datasets
    download_food101()
    build_datasets()


def stage_train():
    print("\n" + "="*60)
    print("  STAGE: Train Models")
    print("="*60)
    from pipeline.train_stage1 import train as train_s1
    from pipeline.train_stage2 import train as train_s2
    print("\n[ Stage 1: Food Classifier ]")
    train_s1()
    print("\n[ Stage 2: Portion Estimator ]")
    train_s2()


def stage_cascade():
    print("\n" + "="*60)
    print("  STAGE: Cascade Pipeline Evaluation")
    print("="*60)
    from pipeline.cascade import evaluate
    evaluate()


def stage_llm(provider: str, limit: int):
    print("\n" + "="*60)
    print("  STAGE: LLM Evaluation")
    print("="*60)
    from evaluation.llm_eval import evaluate
    evaluate(provider=provider, limit=limit)


def stage_compare():
    print("\n" + "="*60)
    print("  STAGE: Final Comparison Figures")
    print("="*60)
    from evaluation.compare import generate_all
    generate_all()


def main():
    parser = argparse.ArgumentParser(
        description="Carbohydrate counting dissertation pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "cascade", "llm", "compare"],
        default="all",
        help="Which stage to run (default: all)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "both"],
        default="anthropic",
        help="LLM provider for evaluation (default: anthropic)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit LLM evaluation to N images. Use --limit 25 for a cheap test."
    )
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  AI Carbohydrate Counting — Dissertation Pipeline")
    print("  Jake Richardson-Price")
    print("█"*60)

    if args.stage in ("all", "data"):
        stage_data()

    if args.stage in ("all", "train"):
        stage_train()

    if args.stage in ("all", "cascade"):
        stage_cascade()

    if args.stage in ("all", "llm"):
        stage_llm(args.provider, args.limit)

    if args.stage in ("all", "compare"):
        stage_compare()

    print("\n" + "█"*60)
    print("  Pipeline complete! Check the results/ folder.")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
