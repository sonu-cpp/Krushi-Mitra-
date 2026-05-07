# 🌾 Krushi Mitra
**AI-Enabled Web-Based Agricultural Decision Support System**
BCA Final Year Project · SOA University (ITER), 2026

## Features
- 🌱 Crop Recommendation (RandomForest, 99.32% accuracy)
- 🧪 Fertilizer Recommendation (rule-based on soil NPK)
- 🔬 Plant Disease Detection (CNN / PyTorch)
- 📊 Mandi Price Display (Odisha + national markets)
- 🌐 Multilingual: English, Hindi (हिंदी), Odia (ଓଡ଼ିଆ)

## Setup & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
krushi_mitra/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── models/
│   ├── RandomForest_new.pkl    # Crop recommendation model
│   ├── NBClassifier.pkl
│   ├── SVMClassifier.pkl
│   └── plant_disease_model.pth # Disease detection CNN
├── data/
│   ├── crop_recommendation.csv
│   └── fertilizer.csv
└── utils/
    ├── translations.py     # EN / HI / OD language strings
    ├── crop_utils.py       # Crop prediction logic
    ├── fertilizer_utils.py # Fertilizer advice logic
    └── mandi_utils.py      # Mandi price data
```
