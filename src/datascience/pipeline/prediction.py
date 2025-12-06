import joblib
from pathlib import Path
import pandas as pd

class PredictionPipeline:
    def __init__(self, best_model_path: str = "artifacts/model_trainer/best_model.joblib"):
        self.best_model_path = Path(best_model_path)
        if not self.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {self.best_model_path}")
        
        # Load the best model
        self.model = joblib.load(self.best_model_path)
        print(f"Loaded best model from {self.best_model_path}")

    def predict(self, data: pd.DataFrame):
        """
        Make predictions on the input data.
        
        Args:
            data (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predictions.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame")
        
        prediction = self.model.predict(data)
        return prediction
