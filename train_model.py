"""Notebook ka exact pipeline: duplicates hatao, label-encode, LinearRegression train karo, save karo."""
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)

# LabelEncoder alphabetical order mein encode karta hai (notebook jaisa hi)
ENCODINGS = {
    "sex": {"female": 0, "male": 1},
    "smoker": {"no": 0, "yes": 1},
    "region": {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3},
}
for col, mapping in ENCODINGS.items():
    df[col] = df[col].map(mapping)

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
X, y = df[FEATURES], df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2), "| R2:", round(r2_score(y_test, pred), 4))

joblib.dump({"model": model, "encodings": ENCODINGS, "features": FEATURES}, "insurance_model.joblib")
print("Saved insurance_model.joblib")
