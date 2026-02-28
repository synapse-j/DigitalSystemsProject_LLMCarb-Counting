"""
config.py
==========
All project settings in one place.
Change values here and they apply everywhere automatically.
"""

import os
import torch

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
DATA_DIR         = "data"
FOOD101_DIR      = os.path.join(DATA_DIR, "food101", "food-101")
DATASET_DIR      = os.path.join(DATA_DIR, "carb_dataset")
STAGE1_MODEL_DIR = os.path.join("models", "stage1")
STAGE2_MODEL_DIR = os.path.join("models", "stage2")
RESULTS_DIR      = "results"

# ══════════════════════════════════════════════════════════════════
# TRAINING SETTINGS  ← adjust these for your machine
# ══════════════════════════════════════════════════════════════════
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 32
EPOCHS_S1    = 20     # Stage 1: food classifier
EPOCHS_S2    = 15     # Stage 2: portion estimator
LR           = 1e-4
NUM_WORKERS  = 0      # 0 = safe on Windows; set 4 on Linux/Mac with GPU

# How many images per food category to use
# 200 = fast dev run (~2hrs training)
# 750 = full dissertation run (~6hrs training)
IMAGES_PER_CLASS = 200

# ══════════════════════════════════════════════════════════════════
# CARBOHYDRATE RANGES
# ══════════════════════════════════════════════════════════════════
NUM_CARB_RANGES = 5

CARB_RANGE_LABELS = {
    0: "0-20g",
    1: "21-40g",
    2: "41-60g",
    3: "61-80g",
    4: "81g+",
}

# Portion size classes for Stage 2
PORTION_LABELS   = {0: "small", 1: "medium", 2: "large"}
NUM_PORTIONS     = 3

# Portion multipliers applied to base carb values
# small = 60% of a standard serving, large = 150%
PORTION_MULTIPLIER = {0: 0.6, 1: 1.0, 2: 1.5}


def grams_to_range(grams: int) -> int:
    """Convert carbohydrate grams to range label 0-4."""
    if grams <= 20:   return 0
    elif grams <= 40: return 1
    elif grams <= 60: return 2
    elif grams <= 80: return 3
    else:             return 4


# ══════════════════════════════════════════════════════════════════
# FOOD → CARB LOOKUP TABLE
# Base carb values (grams) per food at a MEDIUM portion.
# Sources: NHS Eatwell Guide, USDA FoodData Central, Diabetes UK.
# ══════════════════════════════════════════════════════════════════
FOOD_BASE_CARBS = {
    "apple_pie":               55,
    "baby_back_ribs":           8,
    "baklava":                 55,
    "beef_carpaccio":           2,
    "beef_tartare":             3,
    "beignets":                50,
    "bibimbap":                35,
    "bread_pudding":           75,
    "breakfast_burrito":       55,
    "bruschetta":              45,
    "caesar_salad":             8,
    "cannoli":                 65,
    "caprese_salad":            6,
    "carrot_cake":             70,
    "ceviche":                  5,
    "cheesecake":              65,
    "cheese_plate":            10,
    "chicken_curry":           25,
    "chicken_quesadilla":      68,
    "chicken_wings":           10,
    "chocolate_cake":          75,
    "chocolate_fondue":        60,
    "chocolate_mousse":        40,
    "churros":                 58,
    "clam_chowder":            18,
    "club_sandwich":           38,
    "crab_cakes":              28,
    "creme_brulee":            35,
    "croque_madame":           35,
    "cup_cakes":               72,
    "deviled_eggs":             1,
    "donuts":                  68,
    "dumplings":               35,
    "edamame":                 10,
    "eggs_benedict":           18,
    "escargots":                2,
    "falafel":                 32,
    "filet_mignon":             0,
    "fish_and_chips":          55,
    "foie_gras":                3,
    "french_fries":            63,
    "french_onion_soup":       20,
    "french_toast":            48,
    "fried_calamari":          20,
    "fried_egg":                1,
    "fried_rice":              45,
    "frozen_yogurt":           30,
    "garlic_bread":            55,
    "gnocchi":                 50,
    "greek_salad":             12,
    "grilled_cheese_sandwich": 35,
    "grilled_salmon":           0,
    "guacamole":               12,
    "gyoza":                   30,
    "hamburger":               35,
    "hot_and_sour_soup":       15,
    "hot_dog":                 25,
    "huevos_rancheros":        30,
    "hummus":                  20,
    "ice_cream":               28,
    "lasagna":                 38,
    "lobster_bisque":          14,
    "lobster_roll_sandwich":   35,
    "macaroni_and_cheese":     55,
    "macarons":                72,
    "miso_soup":                8,
    "mussels":                  6,
    "nachos":                  55,
    "omelette":                10,
    "onion_rings":             40,
    "oysters":                  5,
    "pad_thai":                40,
    "paella":                  52,
    "pancakes":                56,
    "panna_cotta":             30,
    "peking_duck":             10,
    "pho":                     35,
    "pizza":                   55,
    "pork_chop":                0,
    "poutine":                 55,
    "prime_rib":                0,
    "pulled_pork_sandwich":    38,
    "ramen":                   38,
    "red_velvet_cake":         75,
    "risotto":                 55,
    "samosa":                  35,
    "sashimi":                  0,
    "scallops":                 5,
    "seaweed_salad":           10,
    "shrimp_and_grits":        30,
    "spaghetti_bolognese":     55,
    "spaghetti_carbonara":     55,
    "spring_rolls":            30,
    "steak":                    0,
    "strawberry_shortcake":    55,
    "sushi":                   40,
    "tacos":                   28,
    "takoyaki":                35,
    "tiramisu":                40,
    "tuna_tartare":             4,
    "waffles":                 45,
}

NUM_FOOD_CLASSES = len(FOOD_BASE_CARBS)


def lookup_carb_range(food_category: str, portion: int) -> int:
    """
    Combine Stage 1 + Stage 2 predictions into a carb range.
    food_category: string name e.g. "pizza"
    portion:       int 0=small, 1=medium, 2=large
    """
    base     = FOOD_BASE_CARBS.get(food_category, 40)
    adjusted = int(base * PORTION_MULTIPLIER[portion])
    return grams_to_range(adjusted)
