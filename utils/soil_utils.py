"""
soil_utils.py  —  Krushi Mitra v2.0
====================================
Soil detection pipeline with two modes:

  1. CNN Mode (default when model file exists)
     - Uses MobileNetV2 fine-tuned on the Phantom-fs 7-class dataset
     - Classes: Alluvial, Black, Laterite, Red, Yellow, Arid, Mountain
     - Model file: models/soil_cnn.pt  (PyTorch, ~14MB)

  2. HSV Fallback Mode (when CNN model is not found)
     - Original color-scoring approach (6 classes)
     - Kept for backward compatibility and offline use

After detection, estimate_npk() uses the soil_crop_db.csv lookup table
to return both NPK estimates AND direct crop recommendations from the DB,
which are merged with the RandomForest output in crop_utils.py.
"""

import numpy as np
from PIL import Image
import io
import os
import csv

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR        = os.path.dirname(__file__)
CNN_MODEL_PATH   = os.path.join(_BASE_DIR, '..', 'models', 'soil_cnn.pt')
SOIL_DB_PATH     = os.path.join(_BASE_DIR, '..', 'data', 'soil_crop_db.csv')

# ── CNN class order (must match training label encoding) ─────────────────────
CNN_CLASSES = ["Alluvial", "Arid", "Black", "Laterite", "Mountain", "Red", "Yellow"]
# Maps CNN output label → internal soil key used in SOIL_TYPES dict
CNN_TO_INTERNAL = {
    "Alluvial":  "Alluvial Soil",
    "Arid":      "Arid Soil",
    "Black":     "Black Soil",
    "Laterite":  "Laterite Soil",
    "Mountain":  "Mountain Soil",
    "Red":       "Red Soil",
    "Yellow":    "Yellow Soil",
}

# ── Soil Type Definitions ─────────────────────────────────────────────────────
SOIL_TYPES = {
    "Red Soil": {
        "description": "Iron-rich, well-drained soil common in Odisha, Jharkhand, AP",
        "color_hint": "#c1440e",
        "npk": {"N": (20, 60), "P": (20, 50), "K": (30, 80)},
        "ph": (5.5, 6.5),
        "characteristics": "Low nitrogen, good potassium, well-drained",
        "suitable_crops": ["groundnut", "cotton", "maize", "rice", "mango"],
    },
    "Black Soil": {
        "description": "Clay-rich, moisture-retentive soil common in Maharashtra, MP",
        "color_hint": "#2c2c2c",
        "npk": {"N": (40, 90), "P": (30, 70), "K": (80, 150)},
        "ph": (6.5, 8.0),
        "characteristics": "High potassium, moderate nitrogen, moisture retentive",
        "suitable_crops": ["cotton", "soybean", "wheat", "chickpea", "sugarcane"],
    },
    "Alluvial Soil": {
        "description": "Fertile river-deposited soil, most common in Indo-Gangetic plains",
        "color_hint": "#c2a06e",
        "npk": {"N": (60, 120), "P": (40, 90), "K": (60, 130)},
        "ph": (6.0, 7.5),
        "characteristics": "Well-balanced NPK, very fertile, suitable for most crops",
        "suitable_crops": ["rice", "wheat", "maize", "sugarcane", "banana"],
    },
    "Laterite Soil": {
        "description": "Leached, acidic soil found in Odisha hills, Kerala, Karnataka",
        "color_hint": "#b5651d",
        "npk": {"N": (15, 45), "P": (10, 35), "K": (20, 60)},
        "ph": (4.5, 6.0),
        "characteristics": "Low fertility, acidic, needs heavy fertilization",
        "suitable_crops": ["cashew", "coconut", "tea", "coffee", "rubber"],
    },
    "Sandy Soil": {
        "description": "Coarse, fast-draining soil with low water retention",
        "color_hint": "#e8d5a3",
        "npk": {"N": (10, 35), "P": (10, 30), "K": (20, 50)},
        "ph": (5.5, 7.0),
        "characteristics": "Low fertility, drains fast, needs frequent irrigation",
        "suitable_crops": ["watermelon", "muskmelon", "groundnut", "carrot"],
    },
    "Loamy Soil": {
        "description": "Balanced mix of sand, silt and clay — ideal for agriculture",
        "color_hint": "#8b6914",
        "npk": {"N": (60, 130), "P": (50, 100), "K": (70, 140)},
        "ph": (6.0, 7.0),
        "characteristics": "Best all-round soil, balanced drainage and fertility",
        "suitable_crops": ["rice", "wheat", "vegetables", "maize", "cotton"],
    },
    # ── New CNN classes ────────────────────────────────────────────────────────
    "Yellow Soil": {
        "description": "Similar to Red soil with slightly higher organic matter, eastern India",
        "color_hint": "#d4a843",
        "npk": {"N": (20, 55), "P": (15, 45), "K": (25, 70)},
        "ph": (5.5, 7.0),
        "characteristics": "Moderate fertility, similar to Red soil, found in Odisha",
        "suitable_crops": ["rice", "maize", "groundnut", "pigeonpeas", "mungbean"],
    },
    "Arid Soil": {
        "description": "Dry, low organic matter soil with saline/alkaline tendency",
        "color_hint": "#e8c99a",
        "npk": {"N": (10, 40), "P": (10, 35), "K": (20, 60)},
        "ph": (7.5, 9.0),
        "characteristics": "Low organic matter, drought conditions, Rajasthan/Gujarat",
        "suitable_crops": ["barley", "mothbeans", "pomegranate", "mungbean"],
    },
    "Mountain Soil": {
        "description": "High organic matter, cool temperature soil found in hills/NE India",
        "color_hint": "#6b7c4a",
        "npk": {"N": (30, 90), "P": (20, 60), "K": (30, 80)},
        "ph": (5.0, 7.0),
        "characteristics": "Rich organic matter, cool climate, altitude-specific",
        "suitable_crops": ["apple", "tea", "potato", "ginger", "blackgram"],
    },
}

# ── Drainage → rainfall shift (mm) ───────────────────────────────────────────
DRAINAGE_RAINFALL_SHIFT = {
    "Drains very fast (within minutes)": -120,
    "Drains normally (few hours)":           0,
    "Stays wet for a long time":          +140,
}

# ── Rainfall label → base rainfall mm ────────────────────────────────────────
RAINFALL_TO_MM = {
    "Low (dry region, <600mm/year)":             55,
    "Moderate (600-1200mm/year)":               110,
    "High (>1200mm/year, like Odisha coast)":   200,
}

# ── Previous crop → NPK shift ─────────────────────────────────────────────────
PREV_CROP_MODIFIER = {
    "Legume (dal, soybean, groundnut)":     {"N": +80,  "P": +18, "K":   0},
    "Cereal (rice, wheat, maize)":          {"N": +58,  "P":  +1, "K": +10},
    "Vegetable":                            {"N": +29,  "P": +32, "K": +20},
    "Cash crop (cotton, sugarcane)":        {"N":  +3,  "P": +106,"K": +170},
    "This is my first crop / Not sure":     {"N":   0,  "P":   0, "K":   0},
}

# ── Soil base NPK ─────────────────────────────────────────────────────────────
SOIL_BASE_NPK = {
    "Red Soil":      (20,  27,  30,  5.8,  31,  50),
    "Black Soil":    (40,  46,  80,  7.2,  24,  65),
    "Alluvial Soil": (22,  47,  30,  6.8,  24,  82),
    "Laterite Soil": (21,  17,  31,  5.5,  27,  94),
    "Sandy Soil":    (100, 18,  50,  6.4,  29,  92),
    "Loamy Soil":    (78,  47,  40,  6.7,  25,  79),
    # New CNN classes
    "Yellow Soil":   (22,  22,  35,  6.0,  29,  65),
    "Arid Soil":     (15,  18,  30,  8.0,  34,  25),
    "Mountain Soil": (45,  35,  45,  6.0,  15,  75),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CNN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

_cnn_model = None
_cnn_available = None   # None = not yet checked


def _load_cnn_model():
    """
    Lazy-loads the MobileNetV2 soil classifier.
    Returns the model on success, None if unavailable.
    """
    global _cnn_model, _cnn_available
    if _cnn_available is not None:
        return _cnn_model

    try:
        import torch
        import torchvision.models as models
        import torch.nn as nn

        if not os.path.exists(CNN_MODEL_PATH):
            _cnn_available = False
            return None

        num_classes = len(CNN_CLASSES)
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        device = torch.device("cpu")
        state = torch.load(CNN_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()

        _cnn_model = model
        _cnn_available = True
        return model

    except Exception as e:
        print(f"[soil_utils] CNN model load failed: {e}. Falling back to HSV.")
        _cnn_available = False
        return None


def _preprocess_for_cnn(image_bytes):
    """Resize + normalize image bytes → torch tensor (1, 3, 224, 224)."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)


def _detect_soil_cnn(image_bytes):
    """
    CNN-based soil detection.
    Returns (internal_soil_name, confidence_pct, avg_rgb) or None on failure.
    """
    model = _load_cnn_model()
    if model is None:
        return None

    try:
        import torch
        tensor = _preprocess_for_cnn(image_bytes)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top_idx   = probs.argmax().item()
        top_label = CNN_CLASSES[top_idx]
        confidence = round(probs[top_idx].item() * 100, 1)

        internal_name = CNN_TO_INTERNAL.get(top_label, top_label + " Soil")

        # avg_rgb for display
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img.resize((150, 150)), dtype=np.float32)
        avg_rgb = tuple(int(x) for x in arr.mean(axis=(0, 1)))

        return internal_name, min(confidence, 99.0), avg_rgb

    except Exception as e:
        print(f"[soil_utils] CNN inference error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  HSV FALLBACK CLASSIFIER  (original logic, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_soil_hsv(image_bytes):
    """
    Original HSV color-scoring classifier (6 classes).
    Returns (soil_type_name, confidence_pct, avg_rgb).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((150, 150))
    arr = np.array(img, dtype=np.float32)

    h, w = arr.shape[:2]
    margin = h // 5
    center = arr[margin:h-margin, margin:w-margin]
    avg_rgb = center.mean(axis=(0, 1))
    R, G, B = avg_rgb

    max_c = max(R, G, B)
    min_c = min(R, G, B)
    delta = max_c - min_c

    if delta == 0:
        hue = 0
    elif max_c == R:
        hue = 60 * (((G - B) / delta) % 6)
    elif max_c == G:
        hue = 60 * (((B - R) / delta) + 2)
    else:
        hue = 60 * (((R - G) / delta) + 4)

    saturation = 0 if max_c == 0 else delta / max_c
    value      = max_c / 255.0
    brightness = (R + G + B) / 3

    scores = {}

    red_score = 0
    if (hue < 25 or hue > 340) and saturation > 0.2:
        red_score += 50
    if R > G and R > B:
        red_score += 30
    if 80 < brightness < 180:
        red_score += 20
    scores["Red Soil"] = red_score

    black_score = 0
    if brightness < 80:
        black_score += 70
    if saturation < 0.3:
        black_score += 30
    scores["Black Soil"] = black_score

    alluvial_score = 0
    if 15 < hue < 45 and saturation > 0.15 and brightness > 120:
        alluvial_score += 60
    if abs(R - G) < 40 and R > B:
        alluvial_score += 40
    scores["Alluvial Soil"] = alluvial_score

    laterite_score = 0
    if 10 < hue < 30 and saturation > 0.3 and 80 < brightness < 160:
        laterite_score += 60
    if R > G > B:
        laterite_score += 40
    scores["Laterite Soil"] = laterite_score

    sandy_score = 0
    if brightness > 170 and saturation < 0.35:
        sandy_score += 60
    if 20 < hue < 60 and value > 0.7:
        sandy_score += 40
    scores["Sandy Soil"] = sandy_score

    loamy_score = 0
    if 20 < hue < 40 and 0.2 < saturation < 0.5 and 100 < brightness < 170:
        loamy_score += 60
    if abs(R - G) < 30 and R > 100:
        loamy_score += 40
    scores["Loamy Soil"] = loamy_score

    best  = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf  = round((scores[best] / total) * 100, 1)

    return best, min(conf, 92.0), tuple(int(x) for x in avg_rgb)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC DETECT FUNCTION  (auto-selects CNN or HSV)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_soil_type(image_bytes):
    """
    Detects soil type from an image.
    Tries CNN first; falls back to HSV if model is not available.

    Returns:
        soil_name   (str)   — e.g. "Red Soil"
        confidence  (float) — 0-99 %
        avg_rgb     (tuple) — (R, G, B) for display
        method      (str)   — "CNN" or "HSV"
    """
    cnn_result = _detect_soil_cnn(image_bytes)
    if cnn_result is not None:
        name, conf, rgb = cnn_result
        return name, conf, rgb, "CNN"

    name, conf, rgb = _detect_soil_hsv(image_bytes)
    return name, conf, rgb, "HSV"


# ═══════════════════════════════════════════════════════════════════════════════
#  SOIL CROP DB — lookup from CSV
# ═══════════════════════════════════════════════════════════════════════════════

_soil_db = None   # cached after first load


def _load_soil_db():
    """
    Loads soil_crop_db.csv into a dict keyed by soil_type.
    Tries the data/ folder relative to project root.
    Falls back to the utils/ folder (dev convenience).
    """
    global _soil_db
    if _soil_db is not None:
        return _soil_db

    candidates = [
        SOIL_DB_PATH,
        os.path.join(_BASE_DIR, 'soil_crop_db.csv'),
        os.path.join(_BASE_DIR, '..', 'soil_crop_db.csv'),
    ]

    for path in candidates:
        if os.path.exists(path):
            db = {}
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['soil_type'].strip()
                    row['primary_crops']   = [c.strip() for c in row['primary_crops'].split(',')]
                    row['secondary_crops'] = [c.strip() for c in row['secondary_crops'].split(',')]
                    row['avoid_crops']     = [c.strip() for c in row['avoid_crops'].split(',')]
                    db[key] = row
            _soil_db = db
            return db

    print("[soil_utils] soil_crop_db.csv not found. DB lookup disabled.")
    _soil_db = {}
    return {}


def get_db_crop_recommendations(soil_type):
    """
    Returns direct crop recommendations from the soil DB for a given soil type.

    Args:
        soil_type (str): e.g. "Red Soil", "Black Soil"

    Returns:
        dict with keys: primary_crops, secondary_crops, avoid_crops, notes
        or empty dict if not found.
    """
    db = _load_soil_db()

    # Try exact match first, then strip " Soil" suffix to match CSV keys
    record = db.get(soil_type) or db.get(soil_type.replace(" Soil", ""))
    if not record:
        return {}

    return {
        "primary_crops":   record["primary_crops"],
        "secondary_crops": record["secondary_crops"],
        "avoid_crops":     record["avoid_crops"],
        "notes":           record.get("notes", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NPK ESTIMATOR  (unchanged logic, new soil types added)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_npk(soil_type, drainage, rainfall_label, prev_crop):
    """
    Estimates N, P, K, pH, temperature, humidity, rainfall from
    soil type + 3 farmer questions.
    Now supports 9 soil types (6 original + 3 new CNN classes).
    """
    base = SOIL_BASE_NPK.get(soil_type, SOIL_BASE_NPK["Red Soil"])
    N, P, K, ph, temperature, humidity = base

    mod = PREV_CROP_MODIFIER.get(prev_crop, PREV_CROP_MODIFIER["This is my first crop / Not sure"])
    N += mod["N"]
    P += mod["P"]
    K += mod["K"]

    base_rain  = RAINFALL_TO_MM.get(rainfall_label, 110)
    rain_shift = DRAINAGE_RAINFALL_SHIFT.get(drainage, 0)
    rainfall   = base_rain + rain_shift

    if rainfall < 70:
        humidity = max(humidity - 25, 15)
    elif rainfall > 160:
        humidity = min(humidity + 15, 95)

    if drainage == "Stays wet for a long time":
        temperature = max(temperature - 3, 18)
    elif drainage == "Drains very fast (within minutes)":
        temperature = min(temperature + 3, 38)

    N        = round(max(0,   min(N,   140)), 1)
    P        = round(max(0,   min(P,   145)), 1)
    K        = round(max(0,   min(K,   205)), 1)
    ph       = round(max(3.5, min(ph,  9.0)), 2)
    rainfall = round(max(20,  min(rainfall, 300)), 1)

    return {
        "N":           N,
        "P":           P,
        "K":           K,
        "ph":          ph,
        "temperature": temperature,
        "humidity":    humidity,
        "rainfall":    rainfall,
    }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== CNN availability check ===")
    model = _load_cnn_model()
    print(f"CNN model loaded: {model is not None}")

    print("\n=== Soil DB check ===")
    db = _load_soil_db()
    for soil in ["Red", "Black", "Alluvial", "Laterite", "Yellow", "Arid", "Mountain"]:
        rec = get_db_crop_recommendations(soil)
        print(f"{soil}: primary={rec.get('primary_crops', 'NOT FOUND')}")

    print("\n=== NPK estimates for new soil types ===")
    for soil in ["Yellow Soil", "Arid Soil", "Mountain Soil"]:
        p = estimate_npk(soil, "Drains normally (few hours)",
                         "Moderate (600-1200mm/year)",
                         "This is my first crop / Not sure")
        print(f"{soil}: N={p['N']} P={p['P']} K={p['K']} pH={p['ph']}")
