import sys
import os
import numpy as np
import pandas as pd
import pickle
from Foil_Trees import domain_mappers, contrastive_explanation

# --- Configuration ---
SEED = 42
DATASET_NAME = "diabetes"
DATA_DIR = os.path.join("split_datasets", "npy", DATASET_NAME)
MODEL_PATH = os.path.join("saved_models", f"{DATASET_NAME}_Random_Forest_tuned.pkl")

# --- Load dataset metadata ---
import json
with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]

# --- Load train/test splits from .npy ---
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# --- Load pretrained model ---
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# --- Domain mapping for FOIL ---
dm = domain_mappers.DomainMapperTabular(X_train, feature_names=feature_names)

# --- Explanation setup ---
exp = contrastive_explanation.ContrastiveExplanation(
    domain_mapper=dm,
    regression=False,  # Set to True if you're using regression; otherwise False
    explanator=contrastive_explanation.TreeExplanator(),
    verbose=False
)

# --- Choose sample to explain ---
test_num = 2
sample = X_test[test_num]

# --- Predict and Explain ---
print("=== FOIL Explanation ===")
print("Predicted value:", model.predict([sample])[0])

explanation = exp.explain_instance_domain(model.predict, sample.reshape(1, -1))
print("\nExplanation:")
print(explanation)
