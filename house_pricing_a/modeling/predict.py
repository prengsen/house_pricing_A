from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
import typer

from house_pricing_a.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class InferencePipeline(BaseEstimator, RegressorMixin):
    """Flujo de inferencia sobre datos crudos con un modelo ya entrenado.

    Selecciona `features`, aplica `model` (preprocesamiento + modelo, entrenado sobre el
    target en escala log) y revierte la transformación con `inverse_func`.
    """

    def __init__(self, model, features, inverse_func=np.expm1):
        self.model = model
        self.features = features
        self.inverse_func = inverse_func

    def fit(self, X=None, y=None):
        # El modelo llega entrenado; no se reentrena en inferencia
        return self

    def predict(self, X):
        return self.inverse_func(self.model.predict(X[self.features]))

    def __sklearn_is_fitted__(self):
        return True


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
