import numpy as np
import pandas as pd
import json
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Store original dataset info for metadata
original_shape = df.shape
original_columns = df.columns.tolist()

print(f"Original dataset shape: {original_shape}")
print(f"Original columns: {original_columns}")

# Handling outliers using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df_before_outlier_removal = df.shape[0]
df = df[(z_scores < 3).all(axis=1)]
df_after_outlier_removal = df.shape[0]
outliers_removed = df_before_outlier_removal - df_after_outlier_removal

print(f"Outliers removed: {outliers_removed}")

# Remove duplicates
df_before_duplicate_removal = df.shape[0]
df = df.drop_duplicates()
df_after_duplicate_removal = df.shape[0]
duplicates_removed = df_before_duplicate_removal - df_after_duplicate_removal

print(f"Duplicates removed: {duplicates_removed}")

# Replace zero values with median (for specific columns that shouldn't have zero values)
zero_replacement_columns = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']
zero_replacements = {}

for col in zero_replacement_columns:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        median_value = df[col].median()
        df[col] = df[col].replace(0, median_value)
        zero_replacements[col] = {
            'zero_count': int(zero_count),
            'replacement_value': float(median_value)
        }

# Note: Age typically shouldn't be replaced if it's 0, but keeping as per original code
if 'Age' in df.columns:
    zero_count = (df['Age'] == 0).sum()
    median_value = df['Age'].median()
    df['Age'] = df['Age'].replace(0, median_value)
    zero_replacements['Age'] = {
        'zero_count': int(zero_count),
        'replacement_value': float(median_value)
    }

print(f"Zero value replacements: {zero_replacements}")

# Divide the dataset into independent and dependent variables
X = df.drop(columns='Outcome')
y = df['Outcome']

# Check class distribution before SMOTE
class_distribution_before = y.value_counts().to_dict()
print(f"Class distribution before SMOTE: {class_distribution_before}")

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)  # Added random_state for reproducibility
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after SMOTE
class_distribution_after = pd.Series(y_resampled).value_counts().to_dict()
print(f"Class distribution after SMOTE: {class_distribution_after}")

# Store column names for preprocessing pipeline
data_columns = X_resampled.columns

# Create preprocessing pipeline
numerical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer([
    ('numerical_pipeline', numerical_pipeline, data_columns)
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.30, 
    random_state=20,
    stratify=y_resampled  # Ensure balanced split
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert processed arrays back to DataFrames for easier handling
X_train_df = pd.DataFrame(X_train_processed, columns=data_columns)
X_test_df = pd.DataFrame(X_test_processed, columns=data_columns)
y_train_df = pd.DataFrame(y_train, columns=['Outcome'])
y_test_df = pd.DataFrame(y_test, columns=['Outcome'])

# Create metadata
metadata = {
    'original_dataset': {
        'shape': original_shape,
        'columns': original_columns
    },
    'preprocessing_steps': {
        'outliers_removed': outliers_removed,
        'duplicates_removed': duplicates_removed,
        'zero_value_replacements': zero_replacements,
        'smote_applied': True,
        'class_distribution_before_smote': class_distribution_before,
        'class_distribution_after_smote': class_distribution_after
    },
    'final_dataset': {
        'features': data_columns.tolist(),
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'train_processed_shape': X_train_processed.shape,
        'test_processed_shape': X_test_processed.shape
    },
    'preprocessing_pipeline': {
        'imputer_strategy': 'median',
        'scaler': 'StandardScaler',
        'test_size': 0.30,
        'random_state': 20
    }
}

# Save all files
print("Saving processed data...")

# Save metadata
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save processed datasets
X_train_df.to_csv('X_train.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

print("Files saved successfully:")
print("- metadata.json")
print("- X_train.csv")
print("- X_test.csv") 
print("- y_train.csv")
print("- y_test.csv")

print(f"\nFinal shapes:")
print(f"X_train: {X_train_df.shape}")
print(f"X_test: {X_test_df.shape}")
print(f"y_train: {y_train_df.shape}")
print(f"y_test: {y_test_df.shape}")