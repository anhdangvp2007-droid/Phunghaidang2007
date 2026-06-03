import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from preprocess import FeatureEngineer

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/vietnam_housing_dataset.csv")
print(df["Address"].head(10))
df = df.dropna(subset=["Area", "Price"])

X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])

# =====================
# PIPELINE
# =====================
cat_features = [
    "City",
    "District",
    "Ward",
    "Street",
    "Project",
    "City_District",
    "Ward_District",
    "Street_District"
]

num_features = [
    "Area",
    "Frontage",
    "Access Road",
    "Floors",
    "Bedrooms",
    "Bathrooms",
    "Log_Area",
    "Area_squared",
    "Area_per_Bedroom",
    "Area_per_Bathroom",
    "Area_per_Floor",
    "Bedroom_Bathroom_Ratio",
    "Is_Project"
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ]
)

model = XGBRegressor(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("feature", FeatureEngineer()),
    ("prep", preprocess),
    ("model", model)
])

# =====================
# TRAIN
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
feature_names = pipeline.named_steps[
    "prep"
].get_feature_names_out()

importance = pipeline.named_steps[
    "model"
].feature_importances_

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

imp_df = imp_df.sort_values(
    "Importance",
    ascending=False
)

print("\n===== TOP 30 FEATURES =====")
print(imp_df.head(30))
print("DONE TRAIN")

joblib.dump(pipeline, "models/pipeline.pkl")
# predict trên test
pred_log = pipeline.predict(X_test)

# =====================
# EVALUATION
# =====================

pred = np.expm1(pred_log)
actual = np.expm1(y_test)

r2 = r2_score(actual, pred)

ratio = actual / pred

print("\n===== RESULT =====")
print(f"R2: {r2:.4f}")

print("\n===== BIAS ANALYSIS =====")
print(f"Mean ratio   : {ratio.mean():.4f}")
print(f"Median ratio : {np.median(ratio):.4f}")
print("\n===== NUM FEATURES =====")
print(num_features)
print(
    imp_df[
        imp_df["Feature"].str.contains("Area", case=False)
    ].sort_values(
        "Importance",
        ascending=False
    )
)