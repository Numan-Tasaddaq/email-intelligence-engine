import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# === Step 1: Load Data ===
input_csv = r"File Path"
df = pd.read_csv(input_csv)

# Remove rows with missing data
df = df.dropna(subset=["subject", "body_chunk", "topic_label"])

# Combine subject and body as a single input text feature
df["text"] = df["subject"].fillna("") + " " + df["body_chunk"].fillna("")

# === Step 2: Prepare Features and Labels ===
X = df["text"]
y = df["topic_label"]

# === Step 3: Train-Test Split (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 4: Build and Train ML Pipeline ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

# === Step 5: Evaluate Model ===
y_pred = pipeline.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# === Step 6: Predict on Entire Dataset (Optional Automation) ===
df["topic_predicted"] = pipeline.predict(df["text"])

# === Step 7: Save Result ===
output_path = r"File Path"
df.to_csv(output_path, index=False)
print(f"ML topic predictions saved to: {output_path}")
