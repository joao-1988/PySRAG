import numpy as np
from typing import Dict, List
from scipy.stats import norm
from ._gbm_trainer import GBMTrainer

__all__ = ['LognormalEstimator']

class LognormalEstimator:
    """
    LognormalEstimator: A two-step gradient boosting model to estimate the parameters
    of a lognormal distribution (μ and σ) from tabular data.

    This estimator is useful for modeling strictly positive, skewed response variables
    under the assumption that the logarithm of the response is approximately normally 
    distributed conditional on the input features.

    Attributes
    ----------
    objective : str
        Task type, fixed to 'regression'.
    eval_metric : str
        Evaluation metric for LightGBM training, fixed to 'l2' loss.
    n_estimators : int
        Maximum number of boosting iterations.
    stopping_rounds : int
        Number of rounds with no improvement to trigger early stopping.
    train_size : float
        Proportion of data used for training (rest for validation).
    log_period : int or None
        Frequency of logging evaluation metrics. If None, logging is disabled.
    random_state : int
        Seed for reproducibility.
    model_mu : GBMTrainer
        Model trained to predict the mean (μ) of the log-transformed target.
    model_logsigma2 : GBMTrainer
        Model trained to predict the log of the variance of the log-transformed target.

    Methods
    -------
    fit(X, y, verbose=False, refit=True):
        Trains two LightGBM models to estimate μ and σ of a lognormal distribution from input features.

    predict_parameters(X):
        Returns estimated μ and σ from the trained models for given input features.

    predict_bin_probs(X, min_val, max_val, num_bins):
        Predicts probability distribution over user-defined value bins for each sample in X.

    get_bin_probs_batch(mu, sigma, min_val, max_val, num_bins):
        Vectorized version of bin probability estimation given arrays of μ and σ.

    _generate_bins(min_val, max_val, num_bins):
        Generates bin edges and their logarithmic transformation for use in PDF/CDF calculations.

    _get_bin_probs(mu, sigma, bins):
        Computes probability mass assigned to each bin by the lognormal distribution parameterized by μ and σ.

    Example
    -------
    >>> estimator = LognormalEstimator()
    >>> estimator.fit(X_train, y_train)
    >>> mu, sigma = estimator.predict_parameters(X_test)
    >>> probs, centers = estimator.predict_bin_probs(X_test, min_val=1, max_val=100, num_bins=10)
    """

    def __init__(
        self,
        n_estimators: int = 10000,
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

    def fit(self, X, y, verbose: bool = False, refit: bool = True):
        """
        Fits lognormal distribution parameters using gradient boosting models on log-transformed data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target values (must be strictly positive).
        verbose : bool, optional
            Whether to log LightGBM training output.
        refit : bool, optional
            If True, refit the models on the entire dataset using best iteration.
        """        
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
        """
        Predicts parameters (μ, σ) of the lognormal distribution for each row in X.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        mu : np.ndarray
            Predicted mean of log(y).
        sigma : np.ndarray
            Predicted standard deviation of log(y).
        """
        pred_mu = self.model_mu.model.predict(X)
        pred_log_sigma2 = self.model_logsigma2.model.predict(X)
        pred_sigma = np.sqrt(np.exp(pred_log_sigma2))
        return pred_mu, pred_sigma  

    def _generate_bins(self, min_val: float, max_val: float, num_bins: int) -> Dict[str, List[float]]:
        """
        Generates bins and their logarithmic equivalents for computing probabilities.

        Parameters
        ----------
        min_val : float
            Minimum bin value (> 0).
        max_val : float
            Maximum bin value.
        num_bins : int
            Number of bins.

        Returns
        -------
        bins : dict
            Dictionary with keys: 'bin_lower', 'bin_upper', 'bin_center',
            'log_bin_lower', 'log_bin_upper', 'log_bin_center'.
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
        Computes the probability that a lognormal variable with given μ and σ falls into each bin.

        Parameters
        ----------
        mu : float
            Mean of the log-transformed variable.
        sigma : float
            Standard deviation of the log-transformed variable.
        bins : dict
            Dictionary of bin boundaries in both normal and log space.

        Returns
        -------
        probs : list
            List of probabilities (summing to 1) for each bin.
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
        """
        Vectorized bin probability estimation given arrays of μ and σ.

        Parameters
        ----------
        mu : np.ndarray
            Mean values of log(y).
        sigma : np.ndarray
            Standard deviations of log(y).
        min_val : float
            Minimum value for binning (must be > 0).
        max_val : float
            Maximum value for binning.
        num_bins : int
            Number of bins.

        Returns
        -------
        probs : np.ndarray
            Matrix of shape (n_samples, num_bins) with bin probabilities.
        centers : np.ndarray
            Bin center values in original scale.
        """
        bins = self._generate_bins(min_val, max_val, num_bins)
        probs = [self._get_bin_probs(mu[i], sigma[i], bins) for i in range(len(mu))]
        return np.array(probs), np.array(bins['bin_center'])

    def predict_bin_probs(self, X, min_val: float, max_val: float, num_bins: int):
        """
        Predicts probabilities for each sample falling into user-defined value bins.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        min_val : float
            Minimum value for bin range (must be > 0).
        max_val : float
            Maximum value for bin range.
        num_bins : int
            Number of bins.

        Returns
        -------
        probs : np.ndarray
            Array of shape (n_samples, num_bins) with probabilities.
        centers : np.ndarray
            Bin center values in original (non-log) scale.
        """
        mu, sigma = self.predict_parameters(X)
        return self.get_bin_probs_batch(mu, sigma, min_val, max_val, num_bins)