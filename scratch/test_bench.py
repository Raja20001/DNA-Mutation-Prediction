"""
Test script to verify feature extraction and model accuracy benchmarks.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load raw dataset
df = pd.read_csv("data/raw/synthetic_dna_variants.csv")
print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")
print(f"Label counts:\n{df['label'].value_counts()}")
