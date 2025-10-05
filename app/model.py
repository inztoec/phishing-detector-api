import joblib
import pandas as pd

# --- Load the model and features from disk ---
# This is done once when the app starts, not for every request
try:
    model = joblib.load('app/phishing_detector.pkl')
    features = joblib.load('app/features.pkl')
    print("✅ Model and features loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or features: {e}")
    model = None
    features = None

# --- This function MUST be identical to the one in train.py ---
def extract_features(url):
    """Extracts lexical features from a URL."""
    features_dict = {}
    features_dict['url_length'] = len(url)
    features_dict['has_at_symbol'] = 1 if '@' in url else 0
    features_dict['has_hyphen'] = 1 if '-' in url else 0
    features_dict['num_dots'] = url.count('.')
    
    domain = url.split('/')[0]
    try:
        parts = domain.split('.')
        is_ip = len(parts) == 4 and all(part.isdigit() for part in parts)
        features_dict['has_ip_as_domain'] = 1 if is_ip else 0
    except:
        features_dict['has_ip_as_domain'] = 0

    return features_dict

def predict_phishing(url_to_check: str) -> str:
    """
    Takes a URL string, processes it, and returns a prediction.
    """
    if model is None or features is None:
        return "Model not loaded"

    # 1. Extract features from the new URL
    new_url_features = extract_features(url_to_check)

    # 2. Create a DataFrame from the new features
    # The `features` list we loaded ensures the columns are in the correct order
    new_url_df = pd.DataFrame([new_url_features], columns=features)

    # 3. Make a prediction (0 for 'good', 1 for 'bad')
    prediction = model.predict(new_url_df)

    # 4. Return the human-readable label
    return "phishing" if prediction[0] == 1 else "legitimate"