import os.path

import pandas as pd
import joblib


MODEL_LOGIT = joblib.load(os.path.join("models", "logit", "model", "logit_model.pkl"))


class ClassifyIsBankrupt:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def predict(self):
        df = self.read_file()
        x = df[["Х1", "Х2"]]
        return MODEL_LOGIT.predict(x)



    def read_file(self):
        df = pd.read_csv(self.file_path, encoding="utf-8")
        try:
            df = df.drop(columns="Y")
        except Exception:
            pass
        return df
