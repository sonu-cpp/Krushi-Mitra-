import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fertilizer.csv')

_df = None

def load_data():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        # Normalise string columns
        _df['Crop'] = _df['Crop'].str.strip().str.lower()
        _df['Soil_Type'] = _df['Soil_Type'].str.strip()
    return _df


def get_all_crops():
    """Return sorted unique list of crop names (title-cased for display)."""
    df = load_data()
    return sorted(df['Crop'].str.title().unique().tolist())


def get_soil_types_for_crop(crop: str):
    """Return soil types available for a given crop."""
    df = load_data()
    rows = df[df['Crop'] == crop.strip().lower()]
    return sorted(rows['Soil_Type'].unique().tolist())


def recommend_fertilizer(crop: str, soil_type: str):
    """
    Given only crop name and soil type, return full fertilizer recommendation.
    Returns a dict with all details, or None if not found.
    """
    df = load_data()
    row = df[
        (df['Crop'] == crop.strip().lower()) &
        (df['Soil_Type'].str.lower() == soil_type.strip().lower())
    ]

    # Fallback: first entry for that crop if exact soil match missing
    if row.empty:
        row = df[df['Crop'] == crop.strip().lower()]

    if row.empty:
        return None

    r = row.iloc[0]

    result = {
        "crop": r['Crop'].title(),
        "soil_type": r['Soil_Type'],
        "season": r.get('Season', 'N/A'),
        "ideal_npk": {
            "N": int(r['N']),
            "P": int(r['P']),
            "K": int(r['K']),
            "pH": float(r['pH']),
            "soil_moisture": int(r['soil_moisture']),
        },
        "fertilizer_name": r.get('Fertilizer_Name', 'N/A'),
        "fertilizer_dose": r.get('Fertilizer_Dose', 'N/A'),
        "application_method": r.get('Application_Method', 'N/A'),
    }
    return result

