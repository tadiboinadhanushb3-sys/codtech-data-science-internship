# CODTECH Internship Task-1
# Data Pipeline Development (ETL Process)

# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Step 2: Extract - Load Dataset
print("Loading dataset...")

data = pd.read_csv("titanic.csv")

# Show first rows
print("\nFirst 5 rows of dataset:")
print(data.head())

# Show dataset info
print("\nDataset Information:")
print(data.info())

# Show column names
print("\nColumn names:")
print(data.columns)

# Step 3: Data Cleaning

# Fill missing values in age
data['age'].fillna(data['age'].mean(), inplace=True)

print("\nMissing values handled.")

# Step 4: Drop unnecessary column
data.drop(['name'], axis=1, inplace=True)

print("\nUnnecessary columns removed.")

# Step 5: Encode categorical data
encoder = LabelEncoder()

data['sex'] = encoder.fit_transform(data['sex'])

print("\nCategorical column encoded.")

# Step 6: Feature Scaling
scaler = StandardScaler()

data[['age','fare']] = scaler.fit_transform(data[['age','fare']])

print("\nFeature scaling completed.")

# Step 7: Split dataset into features and target
X = data.drop('survived', axis=1)
y = data['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDataset split into training and testing sets.")

# Step 8: Load - Save processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nProcessed files saved successfully.")

# Final message
print("\nETL Data Pipeline Completed Successfully!")