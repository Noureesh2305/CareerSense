import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Preprocess: Split skills and interests
df['Skills'] = df['Skills'].apply(lambda x: x.split(';'))
df['Interests'] = df['Interests'].apply(lambda x: [x])  # wrap in list

# One-hot encode Skills and Interests
mlb = MultiLabelBinarizer()
interests_encoded = pd.DataFrame(mlb.fit_transform(df['Interests']), columns=mlb.classes_)
skills_encoded = pd.DataFrame(mlb.fit_transform(df['Skills']), columns=mlb.classes_)

# Combine features
features = pd.concat([interests_encoded, skills_encoded, df[['Maths', 'CS', 'English']]], axis=1)

# Target labels
labels = df['Recommended_Career']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model and metadata
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/career_model.pkl")
joblib.dump(features.columns.tolist(), "model/feature_columns.pkl")
joblib.dump(mlb, "model/mlb_encoder.pkl")

print("✅ Model trained and saved in /model folder.")
