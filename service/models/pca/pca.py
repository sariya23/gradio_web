import joblib
import pandas as pd

PCA_MODEL = joblib.load("models/pca/model/pca.pkl")
SCALER = joblib.load("models/pca/meta_data/scaler.pkl")


class PCA:
    def __init__(self, file_path):
        self.file_path = file_path

    def transform(self):
        df = self.read_file()
        x1 = df.drop(columns=["sale price"])
        y1 = df["sale price"]
        X_scaled = SCALER.transform(x1)
        X_pca = PCA_MODEL.transform(X_scaled)
        return X_scaled, X_pca, y1


    def read_file(self):
        df = pd.read_csv(self.file_path, delimiter=";")
        return df