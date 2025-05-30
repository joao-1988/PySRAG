import pandas as pd
import numpy as np
from typing import Tuple

class WeightedClassifier:
    """
    A classifier that adjusts predicted class probabilities using a weighting mechanism,
    such as a density estimation model over a continuous variable (e.g., age).

    Attributes
    ----------
    model_prob : object
        Trained classification model with `predict_proba` and `feature_name_`.
    model_density : object
        Trained density estimation model with `predict_bin_probs`.

    Methods
    -------
    predict_proba(X, range_density, num_bins):
        Returns class probabilities weighted by the distribution of a continuous variable.
    """

    def __init__(self, model_prob, model_density):
        self.model_prob = model_prob
        self.model_density = model_density

        # Extract model features
        self.cols_X = self.model_prob.model.feature_name_
        self.cols_X_density = self.model_density.model_mu.model.feature_name_

        # Identify the variable used in model_prob but not in model_density
        diff = [col for col in self.cols_X if col not in self.cols_X_density]
        if not diff:
            raise ValueError("model_prob must contain a variable not used in model_density.")
        self.var_density = diff[0]

    def predict_proba(
        self,
        X: pd.DataFrame,
        range_density: Tuple[float, float] = (0, 120),
        num_bins: int = 100
    ) -> pd.DataFrame:
        """
        Computes class probabilities weighted by the distribution of a continuous variable (e.g., age).

        Parameters
        ----------
        X : pd.DataFrame
            Input features for both models.
        range_density : tuple of float
            Range (min, max) for binning the continuous variable.
        num_bins : int
            Number of bins for the density estimation.

        Returns
        -------
        pd.DataFrame
            Aggregated class probabilities by each unique configuration in `cols_X_density`.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if num_bins <= 0:
            raise ValueError("num_bins must be a positive integer.")
        if range_density[0] >= range_density[1]:
            raise ValueError("Invalid range_density: min should be less than max.")

        # Get conditioning variables (e.g., all features except age)
        X_density = X[self.cols_X_density].drop_duplicates().reset_index(drop=True)

        # Predict bin probabilities for each group
        bin_probs, centers = self.model_density.predict_bin_probs(
            X_density,
            min_val=range_density[0],
            max_val=range_density[1],
            num_bins=num_bins
        )

        bin_df = pd.DataFrame(bin_probs, columns=centers)
        X_range = X_density.join(bin_df).melt(
            id_vars=self.cols_X_density,
            var_name=self.var_density,
            value_name='VAR_PROB'
        ).astype({self.var_density: float})

        # Predict class probabilities
        probs = self.model_prob.model.predict_proba(X_range[self.cols_X])
        classes = list(self.model_prob.model.classes_)

        # Combine and weight by density
        X_range[classes] = probs * X_range[['VAR_PROB']].values

        # Aggregate to get final probabilities per subgroup
        return X_range[self.cols_X_density + classes].groupby(self.cols_X_density).sum().reset_index()
