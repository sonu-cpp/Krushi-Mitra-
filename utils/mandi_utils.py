import pandas as pd
import random
from datetime import datetime, timedelta

# Simulated mandi price data for Odisha and nearby states
# Structured to mimic Agmarknet data format

MANDI_DATA = [
    # Odisha Mandis
    {"state": "Odisha", "market": "Bhubaneswar", "commodity": "Rice", "variety": "Common", "min_price": 1800, "max_price": 2100, "modal_price": 1950},
    {"state": "Odisha", "market": "Bhubaneswar", "commodity": "Wheat", "variety": "Common", "min_price": 2000, "max_price": 2300, "modal_price": 2150},
    {"state": "Odisha", "market": "Cuttack", "commodity": "Rice", "variety": "Common", "min_price": 1750, "max_price": 2050, "modal_price": 1900},
    {"state": "Odisha", "market": "Cuttack", "commodity": "Maize", "variety": "Common", "min_price": 1400, "max_price": 1700, "modal_price": 1550},
    {"state": "Odisha", "market": "Sambalpur", "commodity": "Rice", "variety": "Fine", "min_price": 2100, "max_price": 2500, "modal_price": 2300},
    {"state": "Odisha", "market": "Sambalpur", "commodity": "Cotton", "variety": "Common", "min_price": 5200, "max_price": 5800, "modal_price": 5500},
    {"state": "Odisha", "market": "Berhampur", "commodity": "Groundnut", "variety": "Common", "min_price": 4500, "max_price": 5200, "modal_price": 4900},
    {"state": "Odisha", "market": "Berhampur", "commodity": "Jute", "variety": "Common", "min_price": 4000, "max_price": 4500, "modal_price": 4250},
    {"state": "Odisha", "market": "Rourkela", "commodity": "Maize", "variety": "Yellow", "min_price": 1500, "max_price": 1800, "modal_price": 1650},
    {"state": "Odisha", "market": "Rourkela", "commodity": "Soyabean", "variety": "Common", "min_price": 3800, "max_price": 4300, "modal_price": 4050},
    {"state": "Odisha", "market": "Puri", "commodity": "Rice", "variety": "Basmati", "min_price": 3500, "max_price": 4200, "modal_price": 3850},
    {"state": "Odisha", "market": "Balasore", "commodity": "Coconut", "variety": "Common", "min_price": 1200, "max_price": 1600, "modal_price": 1400},
    {"state": "Odisha", "market": "Koraput", "commodity": "Turmeric", "variety": "Common", "min_price": 6500, "max_price": 8000, "modal_price": 7200},
    {"state": "Odisha", "market": "Koraput", "commodity": "Ginger", "variety": "Common", "min_price": 2800, "max_price": 3500, "modal_price": 3100},

    # West Bengal Mandis
    {"state": "West Bengal", "market": "Kolkata", "commodity": "Rice", "variety": "Common", "min_price": 1900, "max_price": 2200, "modal_price": 2050},
    {"state": "West Bengal", "market": "Kolkata", "commodity": "Jute", "variety": "Common", "min_price": 4200, "max_price": 4800, "modal_price": 4500},
    {"state": "West Bengal", "market": "Siliguri", "commodity": "Tea", "variety": "Common", "min_price": 12000, "max_price": 18000, "modal_price": 15000},

    # Andhra Pradesh Mandis
    {"state": "Andhra Pradesh", "market": "Guntur", "commodity": "Chilli", "variety": "Red", "min_price": 8000, "max_price": 12000, "modal_price": 10000},
    {"state": "Andhra Pradesh", "market": "Vijayawada", "commodity": "Rice", "variety": "Fine", "min_price": 2200, "max_price": 2600, "modal_price": 2400},

    # Maharashtra Mandis
    {"state": "Maharashtra", "market": "Pune", "commodity": "Grapes", "variety": "Common", "min_price": 3000, "max_price": 4500, "modal_price": 3800},
    {"state": "Maharashtra", "market": "Nagpur", "commodity": "Orange", "variety": "Common", "min_price": 2500, "max_price": 3500, "modal_price": 3000},
]

def get_states():
    return sorted(list(set(d["state"] for d in MANDI_DATA)))

def get_commodities(state=None):
    if state:
        data = [d for d in MANDI_DATA if d["state"] == state]
    else:
        data = MANDI_DATA
    return sorted(list(set(d["commodity"] for d in data)))

def get_prices(state=None, commodity=None):
    data = MANDI_DATA
    if state:
        data = [d for d in data if d["state"] == state]
    if commodity:
        data = [d for d in data if d["commodity"] == commodity]

    # Add simulated "today's date" and small random variation to make it feel live
    today = datetime.now().strftime("%d-%m-%Y")
    result = []
    for row in data:
        variation = random.randint(-50, 50)
        result.append({
            "State": row["state"],
            "Market": row["market"],
            "Commodity": row["commodity"],
            "Variety": row["variety"],
            "Min Price (₹/Qtl)": row["min_price"] + variation,
            "Max Price (₹/Qtl)": row["max_price"] + variation,
            "Modal Price (₹/Qtl)": row["modal_price"] + variation,
            "Date": today,
        })

    return pd.DataFrame(result)
