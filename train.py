import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# --- This is the core of our project: Feature Engineering ---
def extract_features(url):
    """Extracts lexical features from a URL."""
    features = {}
    features['url_length'] = len(url)
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_hyphen'] = 1 if '-' in url else 0
    features['num_dots'] = url.count('.')
    
    # Check if an IP address is used as the domain
    domain = url.split('/')[0]
    try:
        # Check if the domain consists of 4 parts separated by dots, and all parts are digits
        parts = domain.split('.')
        is_ip = len(parts) == 4 and all(part.isdigit() for part in parts)
        features['has_ip_as_domain'] = 1 if is_ip else 0
    except:
        features['has_ip_as_domain'] = 0

    return features

def train_phishing_detector():
    """Trains a model to detect phishing URLs and saves it."""
    print("--- Starting Model Training and Feature Engineering ---")

    # Load data
    df = pd.read_csv('data.csv')

    # Apply our feature extraction to each URL in the dataset
    features_list = df['url'].apply(extract_features).tolist()
    features_df = pd.DataFrame(features_list)
    
    print("Extracted features from URLs:")
    print(features_df.head()) # Print first 5 rows of features

    # Define our features (X) and our target (y)
    X = features_df
    y = df['label'].apply(lambda x: 1 if x == 'bad' else 0) # Convert labels to 0 and 1

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Accuracy on test set: {accuracy:.2f}")

    # Save the trained model to a file inside the 'app' directory
    joblib.dump(model, 'app/phishing_detector.pkl')
    print("Model saved to app/phishing_detector.pkl")
    
    # Save the feature columns, which are needed for prediction
    joblib.dump(list(X.columns), 'app/features.pkl')
    print("Feature list saved to app/features.pkl")


if __name__ == "__main__":
    train_phishing_detector()