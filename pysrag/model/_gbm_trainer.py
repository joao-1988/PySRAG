import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

__all__ = ['GBMTrainer']

class GBMTrainer:
    """
    GBMTrainer: A wrapper for training LightGBM models with support for regression, binary, and multiclass classification.

    This class simplifies the training process with built-in data splitting, early stopping, default hyperparameters,
    and optional refitting on the full dataset using the best number of boosting rounds.

    Attributes
    ----------
    objective : str
        Type of task to solve: 'regression', 'binary', or 'multiclass'.
    eval_metric : str
        Evaluation metric to use. If None, defaults to:
            - 'l2' for regression
            - 'binary_logloss' for binary classification
            - 'multi_logloss' for multiclass classification
    n_estimators : int
        Maximum number of boosting iterations.
    stopping_rounds : int
        Number of rounds with no improvement to trigger early stopping.
    train_size : float
        Proportion of the dataset to use for training (rest is for validation).
    log_period : int or None
        Period to log evaluation metrics during training. If None, logging is disabled.
    random_state : int
        Seed used for reproducibility.
    model : lightgbm.LGBMClassifier or lightgbm.LGBMRegressor
        The trained LightGBM model.
    best_iteration : int
        Number of boosting rounds selected by early stopping.

    Methods
    -------
    fit(X, y, verbose=False, refit=True):
        Train the model using LightGBM with early stopping on a validation split.
        If `refit=True`, retrain the model on the full dataset using the best iteration.

    brier_loss(y_true, y_pred):
        Compute the Brier score loss for probabilistic predictions.

    Examples
    --------
    >>> trainer = GBMTrainer(objective='binary')
    >>> model = trainer.fit(X_train, y_train, verbose=True, refit=True)
    >>> loss_name, score, is_higher_better = trainer.brier_loss(y_test, model.predict_proba(X_test)[:, 1])
    """

    def __init__(self, objective='multiclass', eval_metric=None
                 , n_estimators=5000, stopping_rounds=10, train_size=0.8, log_period=None, random_state=0):
        """
        Initialize the GBMTrainer.

        Parameters
        ----------
        objective : str
            Task type: 'regression', 'binary', or 'multiclass'.
        eval_metric : str or None
            Evaluation metric to use. Defaults to task-specific metrics if None.
        n_estimators : int
            Number of boosting iterations.
        stopping_rounds : int
            Number of rounds with no improvement for early stopping.
        train_size : float
            Proportion of the dataset used for training.
        log_period : int or None
            Frequency of logging during training.
        random_state : int
            Seed for reproducibility.
        """        
        self.objective = objective
        self.n_estimators = n_estimators
        self.stopping_rounds = stopping_rounds
        self.train_size = train_size
        self.log_period = log_period
        self.random_state = random_state

        valid_objectives = ['regression', 'multiclass', 'binary']
        if self.objective not in valid_objectives:
            raise ValueError(f"Objective '{self.objective}' is not supported. Choose from {valid_objectives}.")

        if eval_metric is None:
            self.eval_metric = {
                'regression': 'l2',
                'multiclass': 'multi_logloss',
                'binary': 'binary_logloss'
            }[self.objective]
        else:
            self.eval_metric = eval_metric

        self.model_class = lgb.LGBMRegressor if self.objective == 'regression' else lgb.LGBMClassifier

    def fit(self, X, y, verbose=False, refit = True):
        """
        Fit a LightGBM model with training and validation split, early stopping,
        and optional refitting on the full data using the best iteration.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target vector.
        verbose : bool
            Whether to print logs during training.
        refit : bool
            If True, retrain model on full dataset with best number of iterations.

        Returns
        -------
        model : LGBMClassifier or LGBMRegressor
            Trained LightGBM model.
        """
        if self.model_class == lgb.LGBMClassifier:
            values, counts = np.unique(y, return_counts=True)
            remove_category = values[counts < 2]
            if len(remove_category) > 0:
                remove_index = y.isin(remove_category)
                X = X[~remove_index]
                y = y[~remove_index]
                
        stratify = y if self.model_class == lgb.LGBMClassifier else None
        X_treino, X_eval, y_treino, y_eval = train_test_split(
            X, y, train_size=self.train_size, stratify=stratify, random_state=self.random_state
        )

        self.model = self.model_class(
            boosting_type='gbdt',
            objective=self.objective,
            n_estimators=self.n_estimators,
            num_leaves=10,
            max_depth=3,
            n_jobs=-1,
            learning_rate=0.1,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            min_data_in_bin=10,
            random_state=self.random_state,
            verbose=-1
        )

        eval_set = [(X_treino, y_treino), (X_eval, y_eval)]

        callbacks = []
        if self.log_period is not None:
            callbacks.append(lgb.log_evaluation(period=self.log_period))
        if self.stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.stopping_rounds, first_metric_only=True, verbose=verbose))

        self.model.fit(X_treino, y_treino,
                       eval_set=eval_set,
                       eval_metric=self.eval_metric,
                       callbacks=callbacks)
        
        self.best_iteration = self.model.best_iteration_
        
        if refit:
            self.model = self.model_class(
                boosting_type='gbdt',
                objective=self.objective,
                n_estimators=self.best_iteration,
                num_leaves=10,
                max_depth=3,
                n_jobs=-1,
                learning_rate=0.1,
                min_data_in_leaf=30,
                feature_fraction=0.8,
                min_data_in_bin=10,
                random_state=self.random_state,
                verbose=-1
            )            
            self.model.fit(X, y)

        return self.model

    def brier_loss(self, y_true, y_pred):
        """
        Compute Brier score loss for probabilistic predictions.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted probabilities for the positive class.

        Returns
        -------
        metric_name : str
            The name of the metric: 'brier_score_loss'.
        score : float
            Computed Brier score.
        is_higher_better : bool
            Whether higher score indicates better performance (False for Brier loss).
        """
        is_higher_better = False
        score = brier_score_loss(y_true, y_pred)
        return "brier_score_loss", score, is_higher_better