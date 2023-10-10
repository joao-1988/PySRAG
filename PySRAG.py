import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import dump, load

class SRAG:

    def __init__(self, filename, path, path_utils):
        self.__filename = filename
        self.__path = path
        self.__path_utils = path_utils
        self.__dt_file = datetime.strptime(self.__filename[-14:-4], '%d-%m-%Y')
        self.__ano_file = int(self.__dt_file.strftime('%Y'))
        self.__sem_file = int(self.__dt_file.strftime('%U'))
        self.__dt_file_sem = self.__get_previous_sunday(self.__dt_file)
        self.__initialize_attributes()

    def __get_previous_sunday(self, date):
        days_until_sunday = int(date.strftime('%w'))
        previous_sunday = date - timedelta(days=days_until_sunday)
        return previous_sunday

    def __initialize_attributes(self):
        self.__load_common_data()
        self.__load_file_data()

    def __load_common_func(self, filename):
        return load(os.path.join(self.__path_utils, filename))

    def __load_common_data(self):
        self.__regiao_uf = self.__load_common_func('regiao_uf')
        self.__lat_long = self.__load_common_func('lat_long')
        self.__pop_mun = self.__load_common_func('pop_mun')
        self.__uf_loc = self.__load_common_func('uf_loc')
        self.__regiao_loc = self.__load_common_func('regiao_loc')

    def __load_file_data(self):
        col_type = {'DT_NOTIFIC': str
            , 'DT_DIGITA': str
            , 'DT_SIN_PRI': str
            , 'SG_UF_NOT': str
            , 'CO_MUN_NOT': str
            , 'CS_SEXO': str
            , 'NU_IDADE_N': float
            , 'TP_IDADE': float
            , 'AMOSTRA': float
            , 'PCR_RESUL': float
            , 'RES_AN': float
            , 'POS_PCRFLU': float
            , 'TP_FLU_PCR': float
            , 'POS_AN_FLU': float
            , 'TP_FLU_AN': float
            , 'POS_PCROUT': float
            , 'PCR_SARS2': float
            , 'POS_AN_OUT': float
            , 'AN_SARS2': float
            , 'PCR_VSR': float
            , 'AN_VSR': float
            , 'PCR_PARA1': float
            , 'AN_PARA1': float
            , 'PCR_PARA2': float
            , 'AN_PARA2': float
            , 'PCR_PARA3': float
            , 'AN_PARA3': float
            , 'PCR_PARA4': float
            , 'PCR_ADENO': float
            , 'AN_ADENO': float
            , 'PCR_METAP': float
            , 'PCR_BOCA': float
            , 'PCR_RINO': float
            , 'PCR_OUTRO': float
            , 'AN_OUTRO': float
                    }
        cols = list(col_type.keys())
        path_file = os.path.join(self.__path, self.__filename)
        self.__data = (pd.read_csv(path_file, sep=';', encoding='latin-1', engine='pyarrow'
                                   , usecols=cols, dtype=col_type)
                       .query(' (AMOSTRA == 1) & ( (PCR_RESUL == 1) | (RES_AN == 1) ) ')
                       .assign(
            DT_FILE=self.__dt_file
            , ANO_FILE=self.__ano_file
            , SEM_FILE=self.__sem_file
            , DT_FILE_SEM=self.__dt_file_sem
            , DT_SIN_PRI=lambda x: pd.to_datetime(x['DT_SIN_PRI'], format='%d/%m/%Y')
            , ANO_SIN_PRI=lambda x: (x['DT_SIN_PRI'].dt.strftime('%Y')).astype(int)
            , SEM_SIN_PRI=lambda x: (x['DT_SIN_PRI'].dt.strftime('%U')).astype(int)
            , DT_SIN_PRI_SEM=lambda x: x['DT_SIN_PRI'].apply(self.__get_previous_sunday)
            , DIF_SEM_FILE_SIN_PRI=lambda x: ((x['DT_FILE_SEM'] - x['DT_SIN_PRI_SEM']).dt.days / 7).astype(int)
            , IDADE_ANO=lambda x: np.where(x['TP_IDADE'] == 3, np.round(x['NU_IDADE_N'], 2),
                                  np.where(x['TP_IDADE'] == 2, np.round(x['NU_IDADE_N'] /  (12), 2),
                                  np.where(x['TP_IDADE'] == 1, np.round(x['NU_IDADE_N'] / (360), 2), None))).astype(float)
            , CO_MUN_NOT=lambda x: (x['CO_MUN_NOT']).astype(int)
            , POS_FLUA=lambda x: np.where(((x['POS_PCRFLU'] == 1) & (x['TP_FLU_PCR'] == 1)) |
                                          ((x['POS_AN_FLU'] == 1) & (x['TP_FLU_AN'] == 1))
                                          , 1, 0)
            , POS_FLUB=lambda x: np.where(((x['POS_PCRFLU'] == 1) & (x['TP_FLU_PCR'] == 2)) |
                                          ((x['POS_AN_FLU'] == 1) & (x['TP_FLU_AN'] == 2))
                                          , 1, 0)
            , POS_SARS2=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_SARS2'] == 1)) |
                                           ((x['POS_AN_OUT'] == 1) & (x['AN_SARS2'] == 1))
                                           , 1, 0)
            , POS_VSR=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_VSR'] == 1)) |
                                         ((x['POS_AN_OUT'] == 1) & (x['AN_VSR'] == 1))
                                         , 1, 0)
            , POS_PARA1=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_PARA1'] == 1)) |
                                           ((x['POS_AN_OUT'] == 1) & (x['AN_PARA1'] == 1))
                                           , 1, 0)
            , POS_PARA2=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_PARA2'] == 1)) |
                                           ((x['POS_AN_OUT'] == 1) & (x['AN_PARA2'] == 1))
                                           , 1, 0)
            , POS_PARA3=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_PARA3'] == 1)) |
                                           ((x['POS_AN_OUT'] == 1) & (x['AN_PARA3'] == 1))
                                           , 1, 0)
            , POS_PARA4=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_PARA4'] == 1))
                                           , 1, 0)
            , POS_ADENO=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_ADENO'] == 1)) |
                                           ((x['POS_AN_OUT'] == 1) & (x['AN_ADENO'] == 1))
                                           , 1, 0)
            , POS_METAP=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_METAP'] == 1))
                                           , 1, 0)
            , POS_BOCA=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_BOCA'] == 1))
                                          , 1, 0)
            , POS_RINO=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_RINO'] == 1))
                                          , 1, 0)
            , POS_OUTROS=lambda x: np.where(((x['POS_PCROUT'] == 1) & (x['PCR_OUTRO'] == 1)) |
                                            ((x['POS_AN_OUT'] == 1) & (x['AN_OUTRO'] == 1))
                                            , 1, 0)
            , POS_SUM=lambda x: x[['POS_FLUA', 'POS_FLUB', 'POS_SARS2', 'POS_VSR', 'POS_PARA1',
                                   'POS_PARA2', 'POS_PARA3', 'POS_PARA4', 'POS_ADENO', 'POS_METAP',
                                   'POS_BOCA', 'POS_RINO', 'POS_OUTROS']].sum(axis=1)
        )
                       .merge(self.__regiao_uf[['SG_UF_NOT', 'REGIAO']], how='left', on='SG_UF_NOT')
                       .merge(self.__lat_long, how='left', on='CO_MUN_NOT')
                       .merge(self.__uf_loc, how='left', on='SG_UF_NOT')
                       .merge(self.__regiao_loc, how='left', on='REGIAO')
                       .merge(self.__pop_mun, how='left', on='CO_MUN_NOT')
                       [['DT_FILE', 'ANO_FILE', 'SEM_FILE', 'DT_FILE_SEM'
                , 'DT_SIN_PRI', 'ANO_SIN_PRI', 'SEM_SIN_PRI', 'DT_SIN_PRI_SEM'
                , 'DIF_SEM_FILE_SIN_PRI'
                , 'REGIAO', 'REGIAO_LATITUDE', 'REGIAO_LONGITUDE'
                , 'SG_UF_NOT', 'UF_LATITUDE', 'UF_LONGITUDE'
                , 'CO_MUN_NOT', 'LATITUDE', 'LONGITUDE'
                , 'POPULACAO'
                , 'IDADE_ANO', 'CS_SEXO'
                , 'PCR_RESUL', 'RES_AN'
                , 'POS_FLUA', 'POS_FLUB', 'POS_SARS2', 'POS_VSR'
                , 'POS_PARA1', 'POS_PARA2', 'POS_PARA3', 'POS_PARA4'
                , 'POS_ADENO', 'POS_METAP', 'POS_BOCA'
                , 'POS_RINO', 'POS_OUTROS'
                , 'POS_SUM'
                         ]]
                       .sort_values('DT_SIN_PRI', ascending=False)
                       .reset_index(drop=True))

    def __dummy_to_label(self, srs):
        return np.where(srs == 1, srs.name[4:], None)

    def __remove_none(self, lst):
        return [i for i in lst if i != None]

    def generate_training_data(self, lag = None
                               , objective='binary'
                               , cols_X=['REGIAO_LATITUDE', 'REGIAO_LONGITUDE', 'UF_LATITUDE'
                , 'UF_LONGITUDE', 'LATITUDE', 'LONGITUDE', 'POPULACAO', 'IDADE_ANO']
                               , col_y=['POS_SARS2', 'POS_FLUA', 'POS_FLUB', 'POS_VSR', 'POS_DEMAIS']
                               , demais_virus=['POS_PARA1', 'POS_PARA2', 'POS_PARA3', 'POS_PARA4',
                                               'POS_ADENO', 'POS_METAP', 'POS_BOCA', 'POS_RINO', 'POS_OUTROS']):
        if lag == None:
          data = self.__data
        else:
          data = self.__data.query(f'DIF_SEM_FILE_SIN_PRI == {int(lag)}')

        df = data.query('POS_SUM > 0')
        if demais_virus != '':
              df = df.assign(POS_DEMAIS=lambda x: x[demais_virus].max(axis=1))

        if objective == 'binary':
            df = df.reset_index(drop=True)
            X, y = df[cols_X], df[col_y]
        elif objective == 'multiclass':
            virus = (df[col_y]
                     .apply(self.__dummy_to_label)
                     .apply(self.__remove_none, axis=1))
            df = (df[cols_X]
                  .assign(VIRUS=virus)
                  .explode('VIRUS')
                  .reset_index(drop=True))
            X, y = df[cols_X], df['VIRUS']

        return X, y

    def get_start_day_of_week(self, lag=0):
        start_day_week = self.__dt_file_sem - timedelta(weeks=lag)
        year = int(start_day_week.strftime('%Y'))
        week = int(start_day_week.strftime('%U'))
        return {'lag': lag, 'year': year, 'week': week, 'start_day_week': start_day_week}

    @property
    def filename(self):
        return self.__filename

    @property
    def path(self):
        return self.__path

    @property
    def data(self):
        return self.__data

    @property
    def dt_file(self):
        return self.__dt_file

    @property
    def ano_file(self):
        return self.__ano_file

    @property
    def sem_file(self):
        return self.__sem_file

    @property
    def dt_file_sem(self):
        return self.__dt_file_sem


from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

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
