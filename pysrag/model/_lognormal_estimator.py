import numpy as np
from typing import Dict, List
from scipy.stats import norm
from ._gbm_trainer import GBMTrainer

__all__ = ['LognormalEstimator']

class LognormalEstimator:
    def __init__(
        self,
        n_estimators: int = 5000,
        stopping_rounds: int = 10,
        train_size: float = 0.8,
        log_period: int = None,
        random_state: int = 0
    ):
        self.objective = 'regression'
        self.eval_metric = 'l2'
        self.n_estimators = n_estimators
        self.stopping_rounds = stopping_rounds
        self.train_size = train_size
        self.log_period = log_period
        self.random_state = random_state

    def fit(self, X, y, verbose: bool = False, refit: bool = False):
        """Fits lognormal parameters using gradient boosting models."""

        if np.min(y) <= 0:
            raise ValueError("Target values `y` must be positive for lognormal fitting.")

        # Fit model for μ (mean of log(y))
        log_y = np.log(y)
        self.model_mu = GBMTrainer(
            objective=self.objective,
            eval_metric=self.eval_metric,
            n_estimators=self.n_estimators,
            stopping_rounds=self.stopping_rounds,
            train_size=self.train_size,
            log_period=self.log_period,
            random_state=self.random_state
        )
        self.model_mu.fit(X, log_y, verbose=verbose, refit=refit)

        # Fit model for log(σ²)
        pred_log_y = self.model_mu.model.predict(X)
        log_sigma2 = np.log((log_y - pred_log_y) ** 2)
        self.model_logsigma2 = GBMTrainer(
            objective=self.objective,
            eval_metric=self.eval_metric,
            n_estimators=self.n_estimators,
            stopping_rounds=self.stopping_rounds,
            train_size=self.train_size,
            log_period=self.log_period,
            random_state=self.random_state
        )
        self.model_logsigma2.fit(X, log_sigma2, verbose=verbose, refit=refit)    

    def predict_parameters(self, X):
        """Predicts mu and sigma for a lognormal distribution."""
        pred_mu = self.model_mu.model.predict(X)
        pred_log_sigma2 = self.model_logsigma2.model.predict(X)
        pred_sigma = np.sqrt(np.exp(pred_log_sigma2))
        return pred_mu, pred_sigma  

    def _generate_bins(self, min_val: float, max_val: float, num_bins: int) -> Dict[str, List[float]]:
        """
        Generates bins in original and log space.

        :param min_val: Minimum value (must be > 0)
        :param max_val: Maximum value
        :param num_bins: Number of bins
        :return: Dictionary with bin edges and centers (original and log scale)
        """
        if min_val < 0 or max_val <= 0:
            raise ValueError("Both min_val and max_val must be positive.")
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val.")

        edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_lower = edges[:-1]
        bin_upper = edges[1:]
        bin_center = (bin_lower + bin_upper) / 2

        # Log-transform, handling log(0)
        log_bin_lower = np.full_like(bin_lower, -np.inf)
        mask = bin_lower > 0
        log_bin_lower[mask] = np.log(bin_lower[mask])
        log_bin_upper = np.log(bin_upper)
        log_bin_center = np.log(bin_center)

        return {
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "bin_center": bin_center,
            "log_bin_lower": log_bin_lower,
            "log_bin_upper": log_bin_upper,
            "log_bin_center": log_bin_center
        }

    def _get_bin_probs(self, mu: float, sigma: float, bins: Dict[str, np.ndarray]) -> List[float]:
        """
        Computes probabilities for each bin under lognormal assumption.

        :param mu: Mean of log(y)
        :param sigma: Std deviation of log(y)
        :param bins: Bin grid from `_generate_bins`
        :return: List of probabilities for each bin
        """
        log_lower = bins['log_bin_lower']
        log_upper = bins['log_bin_upper']
        bin_center = bins['bin_center']

        cdf_lower = np.zeros_like(log_lower)
        mask = log_lower > -np.inf
        cdf_lower[mask] = norm.cdf(log_lower[mask], mu, sigma)
        cdf_upper = norm.cdf(log_upper, mu, sigma)

        cdf_min = np.min(cdf_lower)
        cdf_max = np.max(cdf_upper)
        num_bins = len(bin_center)

        if cdf_max == cdf_min:
            return [1.0 / num_bins] * num_bins

        probs = (cdf_upper - cdf_lower) / (cdf_max - cdf_min)
        return probs.tolist()
    
    def get_bin_probs_batch(self, mu: np.ndarray, sigma: np.ndarray, min_val: float, max_val: float, num_bins: int):
        """Vectorized version of `_get_bin_probs`."""
        bins = self._generate_bins(min_val, max_val, num_bins)
        probs = [self._get_bin_probs(mu[i], sigma[i], bins) for i in range(len(mu))]
        return np.array(probs), np.array(bins['bin_center'])

    def predict_bin_probs(self, X, min_val: float, max_val: float, num_bins: int):
        """Predict bin probabilities from input features."""
        mu, sigma = self.predict_parameters(X)
        return self.get_bin_probs_batch(mu, sigma, min_val, max_val, num_bins)