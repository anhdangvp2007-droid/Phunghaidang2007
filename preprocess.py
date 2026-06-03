import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # ===== ADDRESS =====
        def safe_split(addr, idx):
            try:
                return str(addr).split(",")[-idx].strip()
            except:
                return "Unknown"
        def extract_street(address):
            try:
                return str(address).split(",")[0].strip()
            except:
                return "Unknown"
        def extract_project(address):
            address = str(address)

            if "Dự án" in address:
                return address.split(",")[0].strip()

            return "No_Project"
        X["District"] = X["Address"].apply(lambda x: safe_split(x, 2))
        X["City"] = X["Address"].apply(lambda x: safe_split(x, 1))
        X["Ward"] = X["Address"].apply(lambda x: safe_split(x, 3))
        X["Ward_District"] = X["Ward"] + "_" + X["District"]
                # =====================
        # STREET FEATURE
        # =====================

        X["Street"] = X["Address"].apply(
            lambda x: str(x).split(",")[0].strip()
        )

        top_street = X["Street"].value_counts().head(100).index

        X["Street"] = X["Street"].apply(
            lambda x: x if x in top_street else "Other"
        )
        print(X["Street"].value_counts().head(4))
        X["Street_District"] = X["Street"] + "_" + X["District"]
        X["City_District"] = X["City"] + "_" + X["District"]
        X["Project"] = X["Address"].apply(extract_project)
        top_project = X["Project"].value_counts().head(50).index

        X["Project"] = X["Project"].apply(
            lambda x: x if x in top_project else "Other"
        )
        X["Is_Project"] = (
    X["Address"]
    .str.contains("Dự án", case=False, na=False)
    .astype(int)
)
        # ===== numeric fill =====
        for c in ["Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]:
            if c in X.columns:
                X[c] = X[c].fillna(0)

        # ===== feature engineering =====
        X["Log_Area"] = np.log1p(X["Area"])
        X["Area_squared"] = X["Area"] ** 2
        X["Area_per_Bedroom"] = X["Area"] / (X["Bedrooms"] + 1)
        X["Area_per_Bathroom"] = X["Area"] / (X["Bathrooms"] + 1)
        X["Area_per_Floor"] = X["Area"] / (X["Floors"] + 1)
        X["Price_per_m2_hint"] = X["Area"]
        X["Bedroom_Bathroom_Ratio"] = (
            X["Bedrooms"] /
            (X["Bathrooms"] + 1)
        )

        X["Area_per_Room"] = (
            X["Area"] /
            (X["Bedrooms"] + X["Bathrooms"] + 1)
        )
        X["Log_Area"] = np.log1p(X["Area"])
        X["Area_squared"] = X["Area"] ** 2
        X["Area_bucket"] = pd.cut(
            X["Area"],
            bins=[0, 30, 50, 80, 120, 200, 1000],
            labels=[1,2,3,4,5,6]
        )
        return X
