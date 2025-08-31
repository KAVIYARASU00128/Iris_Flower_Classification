import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Create static folder if not exists
os.makedirs("static", exist_ok=True)

# Load dataset from CSV
data = pd.read_csv("Iris.csv")

# Drop ID column if exists
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# Features (X) and Target (y)
X = data.drop("Species", axis=1)
y = data["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "iris_model.pkl")
print("✅ Model trained on CSV and saved as iris_model.pkl")

# Save accuracy in a file (so Flask can read it later)
with open("static/metrics.txt", "w") as f:
    f.write(f"Model Accuracy: {acc*100:.2f}%")

# ---------------------- VISUALIZATIONS ----------------------

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("static/confusion_matrix.png")
plt.close()

# 2. Feature Importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig("static/feature_importance.png")
plt.close()

# 3. Pairplot
sns.pairplot(data, hue="Species")
plt.savefig("static/pairplot.png")
plt.close()

print("📊 Plots & metrics saved in static/ folder")
