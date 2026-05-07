import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'RandomForest_new.pkl')

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    model = load_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(max(probabilities) * 100, 2)
    return prediction, confidence


def predict_top_crops(N, P, K, temperature, humidity, ph, rainfall, top_n=3, threshold=8.0):
    model = load_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

    results = []
    for i, (crop, prob) in enumerate(ranked[:top_n]):
        pct = round(prob * 100, 1)
        if i == 0 or pct >= threshold:
            results.append((crop, pct))

    return results

def predict_top_crops_merged(N, P, K, temperature, humidity, ph, rainfall,
                              soil_type=None, top_n=3, threshold=5.0):
    model = load_model()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

    rf_scores = {}
    for i, (crop, prob) in enumerate(ranked):
        pct = round(prob * 100, 1)
        if i == 0 or pct >= threshold:
            rf_scores[crop.lower()] = pct

    db_primary   = []
    db_secondary = []

    if soil_type:
        try:
            from soil_utils import get_db_crop_recommendations
            rec = get_db_crop_recommendations(soil_type)
            db_primary   = [c.lower() for c in rec.get("primary_crops",   [])]
            db_secondary = [c.lower() for c in rec.get("secondary_crops", [])]
        except Exception as e:
            print(f"[crop_utils] DB lookup failed: {e}")

    merged = dict(rf_scores)

    for crop in list(merged.keys()):
        if crop in db_primary:
            merged[crop] = round(min(merged[crop] + 15.0, 99.0), 1)
        elif crop in db_secondary:
            merged[crop] = round(min(merged[crop] +  8.0, 99.0), 1)

    for crop in db_primary:
        if crop not in merged:
            merged[crop] = 20.0  

    results = []
    for crop, conf in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        in_rf = crop in rf_scores
        in_db = crop in db_primary or crop in db_secondary

        if in_rf and in_db:
            source = "RF+DB"
        elif in_rf:
            source = "RF"
        else:
            source = "DB"

        results.append({
            "crop":       crop,
            "confidence": conf,
            "source":     source,
        })

    return results


def get_soil_db_crops(soil_type):
    try:
        from soil_utils import get_db_crop_recommendations
        return get_db_crop_recommendations(soil_type)
    except Exception:
        return {}


CROP_INFO = {
    "rice":        {"season": "Kharif (Jun–Nov)",  "water": "High",   "tip": "Requires waterlogged conditions and warm temperatures."},
    "maize":       {"season": "Kharif / Rabi",     "water": "Medium", "tip": "Well-drained loamy soil works best."},
    "chickpea":    {"season": "Rabi (Oct–Mar)",    "water": "Low",    "tip": "Drought tolerant, ideal for dry regions."},
    "kidneybeans": {"season": "Kharif",            "water": "Medium", "tip": "Prefers cool temperatures and well-drained soil."},
    "pigeonpeas":  {"season": "Kharif",            "water": "Low",    "tip": "Drought-resistant legume, good for intercropping."},
    "mothbeans":   {"season": "Kharif",            "water": "Low",    "tip": "Highly drought tolerant, grows in sandy soils."},
    "mungbean":    {"season": "Kharif / Zaid",     "water": "Low",    "tip": "Short-duration crop, suitable for multiple seasons."},
    "blackgram":   {"season": "Kharif",            "water": "Low",    "tip": "Thrives in humid tropical conditions."},
    "lentil":      {"season": "Rabi (Oct–Apr)",    "water": "Low",    "tip": "Cool-season crop, sensitive to frost."},
    "pomegranate": {"season": "Perennial",         "water": "Low",    "tip": "Drought hardy, grows well in semi-arid regions."},
    "banana":      {"season": "Year-round",        "water": "High",   "tip": "Needs rich soil and consistent moisture."},
    "mango":       {"season": "Summer fruit",      "water": "Medium", "tip": "Dry winters improve fruit quality."},
    "grapes":      {"season": "Perennial",         "water": "Medium", "tip": "Well-drained sandy loam is ideal."},
    "watermelon":  {"season": "Summer (Mar–Jun)",  "water": "Medium", "tip": "Needs long warm days and sandy soil."},
    "muskmelon":   {"season": "Summer",            "water": "Medium", "tip": "Warm temperatures and dry conditions at harvest."},
    "apple":       {"season": "Temperate",         "water": "Medium", "tip": "Requires chilling hours; best in hilly regions."},
    "orange":      {"season": "Winter fruit",      "water": "Medium", "tip": "Subtropical climate with mild winters is ideal."},
    "papaya":      {"season": "Year-round",        "water": "Medium", "tip": "Cannot tolerate frost; grows fast in tropics."},
    "coconut":     {"season": "Perennial",         "water": "High",   "tip": "Coastal sandy soil with high humidity is ideal."},
    "cotton":      {"season": "Kharif",            "water": "Medium", "tip": "Long dry spell at harvest improves fiber quality."},
    "jute":        {"season": "Kharif",            "water": "High",   "tip": "Alluvial soil and high rainfall needed."},
    "coffee":      {"season": "Perennial",         "water": "Medium", "tip": "Shade-grown at altitudes of 600–2000m is best."},
    "groundnut":   {"season": "Kharif",            "water": "Low",    "tip": "Well-drained sandy loam or red soil is ideal."},
    "sugarcane":   {"season": "Year-round",        "water": "High",   "tip": "Deep, well-drained fertile soil needed."},
    "wheat":       {"season": "Rabi (Oct–Mar)",    "water": "Medium", "tip": "Cool dry climate; sown after monsoon retreat."},
    "barley":      {"season": "Rabi",              "water": "Low",    "tip": "Drought tolerant; one of the hardiest cereals."},
    "tea":         {"season": "Perennial",         "water": "High",   "tip": "Acidic well-drained soil at altitude is ideal."},
    "cashew":      {"season": "Perennial",         "water": "Low",    "tip": "Coastal laterite soil with low humidity at harvest."},
    "rubber":      {"season": "Perennial",         "water": "High",   "tip": "Tropical climate with high rainfall needed."},
    "ginger":      {"season": "Kharif",            "water": "Medium", "tip": "Prefers partial shade and well-drained loamy soil."},
    "potato":      {"season": "Rabi / Zaid",       "water": "Medium", "tip": "Cool climate and loose, well-drained soil."},
}


def get_crop_info(crop_name):
    return CROP_INFO.get(crop_name.lower(), {
        "season": "Varies by region",
        "water":  "Moderate",
        "tip":    "Consult your local agricultural extension officer for best practices."
    })
