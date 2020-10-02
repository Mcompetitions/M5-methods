import numpy as np
import pandas as pd
import logging
import datetime
import lightgbm as lgb
import random
import os
import psutil
import argparse


def get_logger(log_name, log_dir_path, log_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path = str(log_dir_path / (log_name + datetime.datetime.today().strftime('_%Y_%m%d.log')))
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(log_level)
    return logger


class Log(object):
    def __init__(self, logger):
        self.logger = logger

    def info(self, *messages):
        return self.logger.info(Log.format_message(messages))

    def debug(self, *messages):
        return self.logger.debug(Log.format_message(messages))

    def warning(self, *messages):
        return self.logger.warning(Log.format_message(messages))

    def error(self, *messages):
        return self.logger.error(Log.format_message(messages))

    def exception(self, *messages):
        return self.logger.exception(Log.format_message(messages))

    @staticmethod
    def format_message(messages):
        if len(messages) == 1 and isinstance(messages[0], list):
            messages = tuple(messages[0])
        return '\t'.join(map(str, messages))

    def log_evaluation(self, period=100, show_stdv=True, level=logging.INFO):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                result = '\t'.join(
                    [lgb.callback._format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
                self.logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))

        _callback.order = 10
        return _callback


class Util(object):
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return

    @staticmethod
    def get_memory_usage():
        return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30, 2)

    @staticmethod
    def reduce_mem_usage(df, verbose=False):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df

    @staticmethod
    def merge_by_concat(df1, df2, merge_on):
        merged_gf = df1[merge_on]
        merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
        new_columns = [col for col in list(merged_gf) if col not in merge_on]
        df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
        return df1


class Params(object):
    def __init__(self, setting):
        self.setting = setting
        self.data_dir_path = Path(setting['data_dir_path'])

        self.raw_dir_path = self.data_dir_path / 'raw'
        self.raw_dir_path.mkdir(parents=True, exist_ok=True)

        self.output_name = Path(setting['output_name'])
        self.output_dir_path = self.data_dir_path / 'output' / self.output_name
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.log_name = setting['log_name']
        self.log_dir_path = self.output_dir_path / 'log'
        self.log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log = Log(get_logger(self.log_name, self.log_dir_path))

        self.result_dir_path = self.output_dir_path / 'result'
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        self.work_dir_path = self.output_dir_path / 'work'
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        self.model_dir_path = self.output_dir_path / 'model'
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

        self.seed = 42
        Util.set_seed(self.seed)

        self.sampling_rate = setting['sampling_rate']
        self.export_all_flag = False
        self.recursive_feature_flag = False

        self.target = 'sales'
        self.start_train_day_x = 1

        self.end_train_day_x_list = [int(fold_id) for fold_id in setting['fold_id_list_csv'].split(',')]
        self.end_train_day_default = 1941

        for end_train_day_x in self.end_train_day_x_list:
            (self.result_dir_path / str(end_train_day_x)).mkdir(parents=True, exist_ok=True)
            (self.work_dir_path / str(end_train_day_x)).mkdir(parents=True, exist_ok=True)
            (self.model_dir_path / str(end_train_day_x)).mkdir(parents=True, exist_ok=True)

        self.end_train_day_x = None

        self.prediction_horizon_list = [int(prediction_horizon) for prediction_horizon in
                                        setting['prediction_horizon_list_csv'].split(',')]
        self.prediction_horizon = None
        self.prediction_horizon_prev = None

        self.main_index_list = ['id', 'd']

        self.raw_train_path = self.raw_dir_path / 'sales_train_evaluation.csv'
        self.raw_price_path = self.raw_dir_path / 'sell_prices.csv'
        self.raw_calendar_path = self.raw_dir_path / 'calendar.csv'
        self.raw_submission_path = self.raw_dir_path / 'sample_submission.csv'

        self.remove_features = ['id', 'state_id', 'store_id', 'date', 'wm_yr_wk', 'd', self.target]
        self.enable_features = None
        self.mean_features = [
            'enc_cat_id_mean', 'enc_cat_id_std',
            'enc_dept_id_mean', 'enc_dept_id_std',
            'enc_item_id_mean', 'enc_item_id_std'
        ]

        return

    def update_file_path(self):
        self.grid_base_path = self.work_dir_path / f'grid_base_{self.prediction_horizon}.pkl'
        self.grid_price_path = self.work_dir_path / f'grid_price_{self.prediction_horizon}.pkl'
        self.grid_calendar_path = self.work_dir_path / f'grid_calendar_{self.prediction_horizon}.pkl'
        self.holdout_path = self.result_dir_path / 'holdout.csv'
        self.lag_feature_path = self.work_dir_path / f'lag_feature_{self.prediction_horizon}.pkl'
        self.target_encoding_feature_path = self.work_dir_path / f'target_encoding_{self.prediction_horizon}.pkl'
        self.result_submission_path = self.result_dir_path / 'submission.csv'
        return

    def reset_dir_path(self):
        self.result_dir_path = self.output_dir_path / 'result'
        self.result_dir_path.mkdir(parents=True, exist_ok=True)

        self.work_dir_path = self.output_dir_path / 'work'
        self.work_dir_path.mkdir(parents=True, exist_ok=True)

        self.model_dir_path = self.output_dir_path / 'work'
        self.model_dir_path.mkdir(parents=True, exist_ok=True)
        return

    def update_predict_horizon(self):
        self.update_file_path()
        self.num_lag_day_list = []
        self.num_lag_day = 15
        for col in range(self.prediction_horizon, self.prediction_horizon + self.num_lag_day):
            self.num_lag_day_list.append(col)

        self.num_rolling_day_list = [7, 14, 30, 60, 180]

        self.num_shift_rolling_day_list = []
        for num_shift_day in [1, 7, 14]:
            for num_rolling_day in [7, 14, 30, 60]:
                self.num_shift_rolling_day_list.append([num_shift_day, num_rolling_day])
        return


project_key = 'm5a'
parser = argparse.ArgumentParser(description='{}_main'.format(project_key))
from pathlib import Path

parser.add_argument('-ddp', '--data_dir_path', type=str, default='.')
parser.add_argument('-opn', '--output_name', type=str, default='default')
parser.add_argument('-spr', '--sampling_rate', type=float, default=1.0)
# parser.add_argument('-spr', '--sampling_rate', type=float, default=0.00008)
parser.add_argument('-flc', '--fold_id_list_csv', type=str, default='1941')
# parser.add_argument('-flc', '--fold_id_list_csv', type=str, default='1941,1913,1885,1857,1829,1577')
parser.add_argument('-plc', '--prediction_horizon_list_csv', type=str, default='7,14,21,28')
args = parser.parse_args()
params = Params(
    {
        'data_dir_path': args.data_dir_path,
        'output_name': args.output_name,
        'log_name': '{}_main'.format(project_key),
        'sampling_rate': args.sampling_rate,
        'fold_id_list_csv': args.fold_id_list_csv,
        'prediction_horizon_list_csv': args.prediction_horizon_list_csv,
    })
