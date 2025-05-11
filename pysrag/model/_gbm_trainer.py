import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

__all__ = ['GBMTrainer']

class GBMTrainer:
    def __init__(self, objective='multiclass', eval_metric=None
                 , n_estimators=5000, stopping_rounds=10, train_size=0.8, log_period=None, random_state=0):
        self.objective = objective
        self.n_estimators = n_estimators
        self.stopping_rounds = stopping_rounds
        self.train_size = train_size
        self.log_period = log_period
        self.random_state = random_state

        # Restringir objetivos permitidos
        valid_objectives = ['regression', 'multi_class', 'binary']
        if self.objective not in valid_objectives:
            raise ValueError(f"Objective '{self.objective}' is not supported. Choose from {valid_objectives}.")

        # Escolher a métrica padrão se não for especificada
        if eval_metric is None:
            self.eval_metric = {
                'regression': 'l2',
                'multi_class': 'multi_logloss',
                'binary': 'binary_logloss'
            }[self.objective]
        else:
            self.eval_metric = eval_metric

        # Escolher o modelo correto com base no objetivo
        self.model_class = lgb.LGBMRegressor if self.objective == 'regression' else lgb.LGBMClassifier

    def fit(self, X, y, verbose=False, refit = False):

        # Para classificação, remover categorias com menos de duas ocorrências
        if self.model_class == lgb.LGBMClassifier:
            values, counts = np.unique(y, return_counts=True)
            remove_category = values[counts < 2]
            if len(remove_category) > 0:
                remove_index = y.isin(remove_category)
                X = X[~remove_index]
                y = y[~remove_index]
                
        # Divisão de treino e validação (stratify apenas para classificação)
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
            self.model.fit(X, y, )

        return self.model

    def brier_loss(self, y_true, y_pred):
        is_higher_better = False
        score = brier_score_loss(y_true, y_pred)
        return "brier_score_loss", score, is_higher_better