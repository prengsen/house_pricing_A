from pathlib import Path

from feature_engine.transformation import YeoJohnsonTransformer
from loguru import logger
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import typer

from house_pricing_a.config import PROCESSED_DATA_DIR

app = typer.Typer()


class SkewedYeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """Aplica Yeo-Johnson solo a las variables con |skew| > threshold en los datos de fit."""

    def __init__(self, variables, threshold=0.75):
        self.variables = variables
        self.threshold = threshold

    def fit(self, X, y=None):
        skews = X[self.variables].apply(skew)
        self.variables_sesgadas_ = skews[skews.abs() > self.threshold].index.tolist()
        self.transformer_ = (
            YeoJohnsonTransformer(variables=self.variables_sesgadas_).fit(X)
            if self.variables_sesgadas_
            else None
        )
        return self

    def transform(self, X):
        return self.transformer_.transform(X) if self.transformer_ else X


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
