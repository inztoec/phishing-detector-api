import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
import re  # New import for regular expressions

# Define paths
BASE_DIR = Path(__file__).resolve(strict=True).parent
DATA_DIR = BASE_DIR.parent / "data"
MODEL_DIR = BASE_DIR.parent / "models"

# Ensure the models directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(url):
    """Extracts features from a single URL."""
    # Original features
    count_dots = url.count('.')
    count_hyphens = url.count('-')
    count_slashes = url.count('/')
    count_at = url.count('@')
    url_length = len(url)

    # --- NEW FEATURES ---
    # Feature 6: Presence of '@' symbol
    has_at_symbol = 1 if count_at > 0 else 0

    # Feature 7: URL uses an IP address as the domain
    # This regex checks for a pattern like xxx.xxx.xxx.xxx
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


def train_model():
    """Trains aLogistic Regression model on the phishing dataset."""
    print("Loading data...")
    # Load the datasets
    benign_df = pd.read_csv(DATA_DIR / "benign-urls.csv")
    phishing_df = pd.read_csv(DATA_DIR / "phishing-urls.csv")

    # Add labels
    benign_df['label'] = 0  # 0 for benign
    phishing_df['label'] = 1  # 1 for phishing

    # Combine datasets
    full_df = pd.concat([benign_df, phishing_df], ignore_index=True)

    print("Extracting features...")
    # Extract features from each URL
    features_list = full_df['url'].apply(extract_features).tolist()

    # Convert the list of features into a DataFrame
    features_df = pd.DataFrame.from_records(features_list)

    # Prepare data for modeling
    X = features_df  # Feature data
    y = full_df['label']  # Target labels

    print("Training the model...")
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save the trained model and the feature names
    model_path = MODEL_DIR / "model.joblib"
    features_path = MODEL_DIR / "features.joblib"

    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), features_path)  # Save the list of feature names

    print(f"Model and features saved to {MODEL_DIR}")
    print(f"Training accuracy: {model.score(X, y):.4f}")


if __name__ == "__main__":
    train_model()
