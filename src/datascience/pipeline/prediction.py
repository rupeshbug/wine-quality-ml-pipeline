import joblib
from pathlib import Path
import pandas as pd

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/model_trainer/best_model.joblib"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Best model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(self, data: pd.DataFrame):
        """
        data: pandas DataFrame with the same features used during training
        returns: model predictions as a numpy array
        """
        return self.model.predict(data)
