import joblib
import pandas as pd
from pathlib import Path
import re  # New import for regular expressions

# Define paths
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURES_PATH = MODEL_DIR / "features.joblib"

# Load the trained model and features
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


def extract_features(url):
    """Extracts features from a single URL to match the trained model."""
    # Original features
    count_dots = url.count('.')
    count_hyphens = url.count('-')
    count_slashes = url.count('/')
    count_at = url.count('@')
    url_length = len(url)

    # --- NEW FEATURES (Identical to train.py) ---
    has_at_symbol = 1 if count_at > 0 else 0
    is_ip_address = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0

    return {
        'url_length': url_length,
        'count_dots': count_dots,
        'count_hyphens': count_hyphens,
        'count_slashes': count_slashes,
        'count_at': count_at,
        'has_at_symbol': has_at_symbol,
        'is_ip_address': is_ip_address
    }


def predict_phishing(url: str):
    """
    Predicts if a URL is phishing or benign.
    Returns the prediction and the probability score.
    """
    # Extract features from the input URL
    url_features = extract_features(url)
    
    # Create a DataFrame with the same columns as the training data
    features_df = pd.DataFrame([url_features], columns=features)
    
    # Get prediction and probability
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1] # Probability of being 'phishing'
    
    return {"prediction": "phishing" if prediction == 1 else "benign", "phishing_probability": probability}