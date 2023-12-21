import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

__all__ = ['GBMTrainer']

class GBMTrainer:
    def __init__(self, objective='binary', eval_metric='binary_logloss'
                 , n_estimators=5000, stopping_rounds=10, train_size=0.8, log_period=None, random_state=0):
        self.objective = objective
        self.eval_metric = eval_metric
        self.n_estimators = n_estimators
        self.stopping_rounds = stopping_rounds
        self.train_size = train_size
        self.log_period = log_period
        self.random_state = random_state

    def fit(self, X, y, verbose=False):

        values, counts = np.unique(y, return_counts=True)
        remove_category = values[counts < 2]
        if len(remove_category) > 0:
            remove_index = y.isin(remove_category)
            X = X[~remove_index]
            y = y[~remove_index]

        if (self.objective == 'multiclass') and (y.nunique() == 2):
            self.eval_metric = ''

        X_treino, X_eval, y_treino, y_eval = train_test_split(X, y, train_size=self.train_size
                                                              , stratify=y, random_state=self.random_state)

        self.model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective=self.objective,
            n_estimators=self.n_estimators,
            num_leaves=10,
            max_depth=3,
            n_jobs=0,
            learning_rate=0.1,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            min_data_in_bin=10,
            random_state=self.random_state,
            verbose=-1
        )

        eval_set = [(X_treino, y_treino), (X_eval, y_eval)]

        callbacks = []
        if self.log_period != None:
            log_evaluation = lgb.log_evaluation(period=self.log_period)
            callbacks.append(log_evaluation)
        if self.stopping_rounds != None:
            early_stopping = lgb.early_stopping(stopping_rounds=self.stopping_rounds, first_metric_only=True,
                                                verbose=verbose)
            callbacks.append(early_stopping)

        self.model.fit(X_treino, y_treino,
                       eval_set=eval_set,
                       eval_metric=self.eval_metric,
                       callbacks=callbacks)

        return self.model

    def brier_loss(self, y_true, y_pred):
        is_higher_better = False
        score = brier_score_loss(y_true, y_pred)
        return "brier_score_loss", score, is_higher_better