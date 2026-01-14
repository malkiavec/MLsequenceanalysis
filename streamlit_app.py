import streamlit as st

st.title("🎈 MLsequenceanalysis")
st.write(
import requests
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# =============================
# 📌 1. DATA EXTRACTION FROM API
# =============================
API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"

def fetch_lottery_data():
    """Fetch lottery data from the API and extract winning numbers."""
    print("Fetching lottery data...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        
        lottery_results = [
            list(map(int, entry["winning_numbers"].split()))
            for entry in data
        ]
        
        print("Data fetched successfully.")
        return lottery_results
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

# =============================
# 📌 2. FEATURE ENGINEERING
# =============================

def sum_based_transformations(lottery_data):
    """Transform lottery data by calculating sums of winning numbers."""
    print("Applying sum-based transformations...")
    transformed_sums = []
    OFFSET_VALUES = [9,10,11]  # Key numbers for transformations

    for i in range(1, len(lottery_data)):
        prev_sum = sum(lottery_data[i - 1])
        curr_sum = sum(lottery_data[i])
        combined_sum = prev_sum + curr_sum

        transformed_sums.append([
            combined_sum + OFFSET_VALUES[0], 
            combined_sum + OFFSET_VALUES[1],
            combined_sum - OFFSET_VALUES[0], 
            combined_sum - OFFSET_VALUES[1]
        ])

    print("Transformations applied successfully.")
    return np.array(transformed_sums)

def digit_frequency_analysis(lottery_data):
    """Analyze digit occurrences for pattern detection."""
    print("Analyzing digit frequencies...")
    digit_counts = Counter()

    for draw in lottery_data:
        for num in draw:
            digit_counts.update(int(digit) for digit in str(num))

    print("Digit frequency analysis completed.")
    return digit_counts

# =============================
# 📌 3. TRANSITIONS ANALYSIS
# =============================

def build_transition_probabilities(lottery_data):
    """Create transition probabilities for number prediction."""
    print("Building transition probabilities...")
    transitions = {}

    for i in range(len(lottery_data) - 1):
        previous = tuple(lottery_data[i])
        next_set = tuple(lottery_data[i + 1])

        transitions.setdefault(previous, Counter())[next_set] += 1

    # Normalize transition probabilities
    for key, counts in transitions.items():
        total = sum(counts.values())
        transitions[key] = {k: v / total for k, v in counts.items()}

    print("Transition probabilities built successfully.")
    return transitions

# =============================
# 📌 4. MACHINE LEARNING MODELS
# =============================

def prepare_ml_data(lottery_data):
    """Prepare data for machine learning models."""
    print("Preparing machine learning data...")
    X = sum_based_transformations(lottery_data)
    y = np.array(lottery_data[1:])  # Shifted labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data prepared for machine learning.")
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    lottery_data = fetch_lottery_data()
    if lottery_data:
        X_train, X_test, y_train, y_test = prepare_ml_data(lottery_data)
        # Continue with model training...
