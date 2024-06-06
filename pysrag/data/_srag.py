import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

__all__ = ['SRAG']

class SRAG:

    def __init__(self, filepath, old_filter=True):
        self.__filepath = filepath
        self.__dt_file = datetime.strptime(self.__filepath[-14:-4], '%d-%m-%Y')
        self.__ano_file = int(self.__dt_file.strftime('%Y'))
        self.__sem_file = int(self.__dt_file.strftime('%U'))
        self.__dt_file_sem = self.__get_previous_sunday(self.__dt_file)
        self.__load_file_data(old_filter)

    def __get_previous_sunday(self, date):
        days_until_sunday = int(date.strftime('%w'))
        previous_sunday = date - timedelta(days=days_until_sunday)
        return previous_sunday

    def __load_file_data(self, old_filter):
        col_type = {'DT_NOTIFIC': str
            , 'DT_DIGITA': str
            , 'DT_SIN_PRI': str
            , 'SG_UF_NOT': str
            , 'ID_MUNICIP': str
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
            , 'CASO_SRAG': float
            , 'HOSPITAL': float
            , 'EVOLUCAO': float
            , 'TOSSE': float
            , 'FEBRE': float
            , 'GARGANTA': float
            , 'DISPNEIA': float
            , 'DESC_RESP': float
            , 'SATURACAO': float
                    }
        
        all_cols = pd.read_csv(self.__filepath, sep=';', encoding='latin-1',  nrows=0).columns
        
        if old_filter:
            col_type.pop('CASO_SRAG')
            data = (pd.read_csv(self.__filepath, sep=';', encoding='latin-1', engine='pyarrow'
                                , usecols=col_type.keys(), dtype=col_type)
                    .query(' (AMOSTRA == 1) & ( (PCR_RESUL == 1) | (RES_AN == 1) ) ') 
                    )  
        elif 'CASO_SRAG' not in all_cols:
            col_type.pop('CASO_SRAG')
            data = (pd.read_csv(self.__filepath, sep=';', encoding='latin-1', engine='pyarrow'
                                , usecols=col_type.keys(), dtype=col_type)
                    .assign(sin_SG = lambda x: np.where( (x['TOSSE'] == 1) | ((x['FEBRE'] == 1) & (x['GARGANTA'] == 1)), 1, 0)
                            ,sin_ad_SRAG = lambda x: np.where( (x['DISPNEIA'] == 1) | (x['DESC_RESP'] == 1) | (x['SATURACAO'] == 1), 1, 0)
                            ,cond_SRAG = lambda x: np.where( ( (x['HOSPITAL'] == 1) | (x['EVOLUCAO'] == 2) ) &
                                                            (x['sin_SG'] == 1) & (x['sin_ad_SRAG'] == 1), 1, 0)
                            )
                    .query('cond_SRAG == 1')
                    )
        else:
            data = (pd.read_csv(self.__filepath, sep=';', encoding='latin-1', engine='pyarrow'
                                , usecols=col_type.keys(), dtype=col_type)
                    .query('CASO_SRAG == 1')
                    )

        self.__data = (data.assign(
            DT_FILE=self.__dt_file
            , ANO_FILE=self.__ano_file
            , SEM_FILE=self.__sem_file
            , DT_FILE_SEM=self.__dt_file_sem
            , DT_SIN_PRI=lambda x: pd.to_datetime(x['DT_SIN_PRI'], format='%d/%m/%Y')
            , ANO_SIN_PRI=lambda x: (x['DT_SIN_PRI'].dt.strftime('%Y')).astype(int)
            , SEM_SIN_PRI=lambda x: (x['DT_SIN_PRI'].dt.strftime('%U')).astype(int)
            , ANO_SEM_SIN_PRI=lambda x: x['ANO_SIN_PRI']*100 + x['SEM_SIN_PRI']
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
                       .merge(self.load_common_data(), how='left', left_on='CO_MUN_NOT', right_on='CD_IBGE')
                       [['DT_FILE', 'ANO_FILE', 'SEM_FILE', 'DT_FILE_SEM'
                , 'DT_SIN_PRI', 'ANO_SIN_PRI', 'SEM_SIN_PRI', 'ANO_SEM_SIN_PRI', 'DT_SIN_PRI_SEM'
                , 'DIF_SEM_FILE_SIN_PRI'
                , 'REGIAO', 'REGIAO_LATITUDE', 'REGIAO_LONGITUDE'
                , 'SG_UF_NOT', 'UF_LATITUDE', 'UF_LONGITUDE'
                , 'ID_MUNICIP', 'CO_MUN_NOT', 'LATITUDE', 'LONGITUDE'
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

    def generate_training_weeks(self, lag = None):
        if lag == None:
          data = self.__data
        else:
          data = self.__data.query(f'DIF_SEM_FILE_SIN_PRI == {int(lag)}')

        df = data[['DT_SIN_PRI_SEM', 'ANO_SIN_PRI', 'SEM_SIN_PRI', 'ANO_SEM_SIN_PRI']].drop_duplicates().reset_index(drop=True)
        
        return df

    def get_start_day_of_week(self, lag=0):
        start_day_week = self.__dt_file_sem - timedelta(weeks=lag)
        year = int(start_day_week.strftime('%Y'))
        week = int(start_day_week.strftime('%U'))
        return {'lag': lag, 'year': year, 'week': week, 'start_day_week': start_day_week}

    @property
    def filepath(self):
        return self.__filepath

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
    
    @staticmethod
    def load_common_data():
        data_path = os.path.abspath(os.path.dirname(__file__))
        data_filepath = os.path.join(data_path, 'common_data.csv')
        return pd.read_csv(data_filepath)