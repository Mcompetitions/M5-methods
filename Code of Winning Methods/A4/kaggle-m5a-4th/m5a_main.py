import time
import warnings
import gc
import pickle
import math
import shutil

from math import ceil
from sklearn.metrics import mean_squared_error

from m5a_eval import WRMSSEEvaluator

warnings.filterwarnings('ignore')


class M5aMain(object):
    def __init__(self, params):
        self.params = params
        self.log = self.params.log
        return

    def load_data(self):
        self.log.info('load_data')
        train_df = pd.read_csv(self.params.raw_train_path)
        self.log.info('train_df.shape', train_df.shape)
        prices_df = pd.read_csv(self.params.raw_price_path)
        self.log.info('prices_df.shape', prices_df.shape)
        calendar_df = pd.read_csv(self.params.raw_calendar_path)
        self.log.info('calendar_df.shape', calendar_df.shape)
        submission_df = pd.read_csv(self.params.raw_submission_path)
        self.log.info('submission_df.shape', submission_df.shape)

        self.log.info('sampling_rate', self.params.sampling_rate)
        if self.params.sampling_rate < 1.0:
            self.log.info('sampling start')
            id_list = train_df['id'].unique().tolist()
            item_id_store_id_list = [
                (id.replace('_evaluation', '')[:-5],
                 id.replace('_evaluation', '')[-4:])
                for id in id_list]
            self.log.info('#id', len(item_id_store_id_list))

            train_sampled_df = pd.DataFrame()
            prices_sampled_df = pd.DataFrame()
            submission_sampled_df = pd.DataFrame()
            num_samples = int(len(item_id_store_id_list) * self.params.sampling_rate)

            self.log.info('#samples', num_samples)
            sample_id_header_list = []
            sample_store_id_list = []
            sample_item_id_list = []
            for sample_index in sorted(np.random.permutation(len(item_id_store_id_list))[:num_samples]):
                sample_item_id = item_id_store_id_list[sample_index][0]
                sample_store_id = item_id_store_id_list[sample_index][1]
                sample_id_header = f'{sample_item_id}_{sample_store_id}'
                self.log.info('sample_index', sample_index, 'sample_id_header', sample_id_header)
                sample_id_header_list.append(sample_id_header)
                sample_store_id_list.append(sample_store_id)
                sample_item_id_list.append(sample_item_id)
            train_sampled_df = pd.concat(
                [train_sampled_df,
                 train_df[train_df['id'].str.contains('|'.join(sample_id_header_list))]])

            prices_sampled_df = pd.concat(
                [prices_sampled_df,
                 prices_df[(prices_df['store_id'].str.contains('|'.join(sample_store_id_list))) & \
                           (prices_df['item_id'].str.contains('|'.join(sample_item_id_list)))]])
            submission_sampled_df = pd.concat(
                [submission_sampled_df,
                 submission_df[submission_df['id'].str.contains('|'.join(sample_id_header_list))]])

            train_df = train_sampled_df.reset_index(drop=True)
            prices_df = prices_sampled_df.reset_index(drop=True)
            submission_df = submission_sampled_df.reset_index(drop=True)
            self.log.info('sampling end')

        self.log.info('train_df.shape', train_df.shape)
        self.log.info('prices_df.shape', prices_df.shape)
        self.log.info('calendar_df.shape', calendar_df.shape)
        self.log.info('submission_df.shape', submission_df.shape)

        return train_df, prices_df, calendar_df, submission_df

    def generate_grid_base(self, train_df, prices_df, calendar_df):
        self.log.info('generate_grid_base')
        self.log.info('melt')
        index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        grid_df = pd.melt(train_df, id_vars=index_columns, var_name='d', value_name=self.params.target)

        self.log.info('grid_df.shape', grid_df.shape)

        self.log.info('remove days before end_train_day_x / generate holdout')
        num_before = grid_df.shape[0]
        grid_df['d_org'] = grid_df['d']
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        holdout_df = grid_df[(grid_df['d'] > self.params.end_train_day_x) & \
                             (grid_df['d'] <= self.params.end_train_day_x + self.params.prediction_horizon)][
            self.params.main_index_list + [self.params.target]
            ]
        holdout_df.to_csv(self.params.holdout_path, index=False)

        grid_df = grid_df[grid_df['d'] <= self.params.end_train_day_x]
        grid_df['d'] = grid_df['d_org']
        grid_df = grid_df.drop('d_org', axis=1)
        num_after = grid_df.shape[0]
        self.log.info(num_before, '-->', num_after)

        self.log.info('add test days')
        add_grid = pd.DataFrame()
        for i in range(self.params.prediction_horizon):
            temp_df = train_df[index_columns]
            temp_df = temp_df.drop_duplicates()
            temp_df['d'] = 'd_' + str(self.params.end_train_day_x + i + 1)
            temp_df[self.params.target] = np.nan
            add_grid = pd.concat([add_grid, temp_df])

        grid_df = pd.concat([grid_df, add_grid])
        grid_df = grid_df.reset_index(drop=True)

        del temp_df, add_grid
        del train_df

        self.log.info('convert to category')
        for col in index_columns:
            grid_df[col] = grid_df[col].astype('category')

        self.log.info('calc release week')
        release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id', 'item_id', 'release']
        grid_df = Util.merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
        del release_df
        grid_df = Util.merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])
        grid_df = grid_df.reset_index(drop=True)

        self.log.info('convert release to int16')
        grid_df['release'] = grid_df['release'] - grid_df['release'].min()
        grid_df['release'] = grid_df['release'].astype(np.int16)

        self.log.info('save grid_base')
        grid_df.to_pickle(self.params.grid_base_path)

        self.log.info('grid_df.shape', grid_df.shape)
        return

    def generate_grid_price(self, prices_df, calendar_df):
        self.log.info('generate_grid_price')
        self.log.info('load grid_base')
        grid_df = pd.read_pickle(self.params.grid_base_path)

        prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
        prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
        prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
        prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
        prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']
        prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
        prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

        calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
        calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
        prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
        del calendar_prices

        prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
            'sell_price'].transform(lambda x: x.shift(1))
        prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
            'sell_price'].transform('mean')
        prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
            'sell_price'].transform('mean')

        prices_df['sell_price_cent'] = [math.modf(p)[0] for p in prices_df['sell_price']]
        prices_df['price_max_cent'] = [math.modf(p)[0] for p in prices_df['price_max']]
        prices_df['price_min_cent'] = [math.modf(p)[0] for p in prices_df['price_min']]

        del prices_df['month'], prices_df['year']

        self.log.info('merge prices')
        original_columns = list(grid_df)
        grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        keep_columns = [col for col in list(grid_df) if col not in original_columns]
        grid_df = grid_df[self.params.main_index_list + keep_columns]
        grid_df = Util.reduce_mem_usage(grid_df)

        self.log.info('save grid_price')
        grid_df.to_pickle(self.params.grid_price_path)
        del prices_df
        return

    def generate_grid_calendar(self, calendar_df):
        self.log.info('generate_grid_calendar')
        grid_df = pd.read_pickle(self.params.grid_base_path)

        grid_df = grid_df[self.params.main_index_list]

        import math, decimal
        dec = decimal.Decimal

        def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
            diff = datetime.datetime.strptime(d, '%Y-%m-%d') - datetime.datetime(2001, 1, 1)
            days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
            lunations = dec("0.20439731") + (days * dec("0.03386319269"))
            phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
            return int(phase_index) & 7

        calendar_df['moon'] = calendar_df.date.apply(get_moon_phase)

        # Merge calendar partly
        icols = ['date',
                 'd',
                 'event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI',
                 'moon',
                 ]

        grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

        icols = ['event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']
        for col in icols:
            grid_df[col] = grid_df[col].astype('category')

        grid_df['date'] = pd.to_datetime(grid_df['date'])

        grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
        grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
        grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['tm_y'] = grid_df['date'].dt.year
        grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
        grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)

        grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
        grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)
        del grid_df['date']

        grid_df.to_pickle(self.params.grid_calendar_path)

        del calendar_df
        del grid_df

        return

    def modify_grid_base(self):
        self.log.info('modify_grid_base')
        grid_df = pd.read_pickle(self.params.grid_base_path)
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

        del grid_df['wm_yr_wk']
        grid_df.to_pickle(self.params.grid_base_path)

        del grid_df
        return

    def generate_grid_full(self, train_df, prices_df, calendar_df):
        self.generate_grid_base(train_df, prices_df, calendar_df)
        self.generate_grid_price(prices_df, calendar_df)
        self.generate_grid_calendar(calendar_df)
        self.modify_grid_base()
        return

    def load_grid_full(self):
        self.log.info('load_grid_full')
        grid_df = pd.concat([pd.read_pickle(self.params.grid_base_path),
                             pd.read_pickle(self.params.grid_price_path).iloc[:, 2:],
                             pd.read_pickle(self.params.grid_calendar_path).iloc[:, 2:]],
                            axis=1)
        return grid_df

    def generate_lag_feature(self):
        self.log.info('generate_lag')
        self.log.info('load gird_base')
        grid_df = pd.read_pickle(self.params.grid_base_path)

        grid_df = grid_df[['id', 'd', 'sales']]

        start_time = time.time()
        self.log.info('create lags')

        grid_df = grid_df.assign(**{
            '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
            for l in self.params.num_lag_day_list
            for col in [self.params.target]
        })

        for col in list(grid_df):
            if 'lag' in col:
                grid_df[col] = grid_df[col].astype(np.float16)

        start_time = time.time()
        self.log.info('create rolling aggs')

        for num_rolling_day in self.params.num_rolling_day_list:
            self.log.info('rolling period', num_rolling_day)
            grid_df['rolling_mean_' + str(num_rolling_day)] = grid_df.groupby(['id'])[self.params.target].transform(
                lambda x: x.shift(self.params.prediction_horizon).rolling(num_rolling_day).mean()).astype(np.float16)
            grid_df['rolling_std_' + str(num_rolling_day)] = grid_df.groupby(['id'])[self.params.target].transform(
                lambda x: x.shift(self.params.prediction_horizon).rolling(num_rolling_day).std()).astype(np.float16)

        if self.params.recursive_feature_flag:
            for num_shift_rolling_day in self.params.num_shift_rolling_day_list:
                num_shift_day = num_shift_rolling_day[0]
                num_rolling_day = num_shift_rolling_day[1]
                col_name = 'rolling_mean_tmp_' + str(num_shift_day) + '_' + str(num_rolling_day)
                grid_df[col_name] = grid_df.groupby(['id'])[self.params.target].transform(
                    lambda x: x.shift(num_shift_day).rolling(num_rolling_day).mean()).astype(np.float16)

        self.log.info('save lag_feature')
        grid_df.to_pickle(self.params.lag_feature_path)

        return

    def generate_target_encoding_feature(self):
        Util.set_seed(self.params.seed)

        grid_df = pd.read_pickle(self.params.grid_base_path)
        grid_df[self.params.target][
            grid_df['d'] > (self.params.end_train_day_x - self.params.prediction_horizon)] = np.nan
        base_cols = list(grid_df)

        icols = [
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        ]

        for col in icols:
            self.log.info('encoding', col)
            col_name = '_' + '_'.join(col) + '_'
            grid_df['enc' + col_name + 'mean'] = grid_df.groupby(col)[self.params.target].transform('mean').astype(
                np.float16)
            grid_df['enc' + col_name + 'std'] = grid_df.groupby(col)[self.params.target].transform('std').astype(
                np.float16)

        keep_cols = [col for col in list(grid_df) if col not in base_cols]
        grid_df = grid_df[['id', 'd'] + keep_cols]

        self.log.info('save target_encoding_feature')
        grid_df.to_pickle(self.params.target_encoding_feature_path)
        return

    def load_grid_by_store(self, store_id):
        df = self.load_grid_full()

        if store_id != 'all':
            df = df[df['store_id'] == store_id]

        df2 = pd.read_pickle(self.params.target_encoding_feature_path)[self.params.mean_features]
        df2 = df2[df2.index.isin(df.index)]

        df3 = pd.read_pickle(self.params.lag_feature_path).iloc[:, 3:]
        df3 = df3[df3.index.isin(df.index)]

        df = pd.concat([df, df2], axis=1)
        del df2

        df = pd.concat([df, df3], axis=1)
        del df3

        enable_features = [col for col in list(df) if col not in self.params.remove_features]
        df = df[['id', 'd', self.params.target] + enable_features]

        df = df[df['d'] >= self.params.start_train_day_x].reset_index(drop=True)

        return df, enable_features

    def load_base_test(self, store_id_set_list):
        base_test = pd.DataFrame()

        for store_id in store_id_set_list:
            temp_df = pd.read_pickle(
                self.params.work_dir_path / f'test_{store_id}_{self.params.prediction_horizon}.pkl')
            temp_df['store_id'] = store_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
        return base_test

    def train_and_predict(self, train_df, calendar_df, prices_df, submission_df):
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,
            'num_leaves': 2 ** 11 - 1,
            'min_data_in_leaf': 2 ** 12 - 1,
            'feature_fraction': 0.5,
            'max_bin': 100,
            'n_estimators': 1400,
            'boost_from_average': False,
            'verbose': -1,
        }

        Util.set_seed(self.params.seed)
        lgb_params['seed'] = self.params.seed

        store_id_set_list = list(train_df['store_id'].unique())

        feature_importance_all_df = pd.DataFrame()
        for store_index, store_id in enumerate(store_id_set_list):
            self.log.info('train', store_id)

            grid_df, enable_features = self.load_grid_by_store(store_id)
            self.params.enable_features = enable_features

            train_mask = grid_df['d'] <= self.params.end_train_day_x
            valid_mask = train_mask & (grid_df['d'] > (self.params.end_train_day_x - self.params.prediction_horizon))
            preds_mask = grid_df['d'] > (self.params.end_train_day_x - 100)

            self.log.info('[{3} - {4}] train {0}/{1} {2}'.format(
                store_index + 1, len(store_id_set_list), store_id,
                self.params.end_train_day_x, self.params.prediction_horizon))
            if self.params.export_all_flag:
                self.log.info('export train')
                grid_df[train_mask].to_csv(
                    self.params.result_dir_path / ('exp_train_' + store_id + '.csv'), index=False)
            train_data = lgb.Dataset(grid_df[train_mask][enable_features],
                                     label=grid_df[train_mask][self.params.target])

            if self.params.export_all_flag:
                self.log.info('export valid')
                grid_df[valid_mask].to_csv(
                    self.params.result_dir_path / ('exp_valid_' + store_id + '.csv'), index=False)
            valid_data = lgb.Dataset(grid_df[valid_mask][enable_features],
                                     label=grid_df[valid_mask][self.params.target])

            if self.params.export_all_flag:
                self.log.info('export test')
                grid_df[preds_mask].to_csv(
                    self.params.result_dir_path / ('exp_test_' + store_id + '.csv'), index=False)

            if self.params.export_all_flag:
                self.log.info('export train_valid_test')
                grid_df[train_mask | valid_mask | preds_mask].to_csv(
                    self.params.result_dir_path / ('exp_train_valid_test_' + store_id + '.csv'), index=False)
            valid_data = lgb.Dataset(grid_df[valid_mask][enable_features],
                                     label=grid_df[valid_mask][self.params.target])

            # Saving part of the dataset for later predictions
            # Removing features that we need to calculate recursively
            grid_df = grid_df[preds_mask].reset_index(drop=True)
            if self.params.recursive_feature_flag:
                keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
                grid_df = grid_df[keep_cols]
            grid_df.to_pickle(self.params.work_dir_path / f'test_{store_id}_{self.params.prediction_horizon}.pkl')
            del grid_df

            Util.set_seed(self.params.seed)
            estimator = lgb.train(lgb_params,
                                  train_data,
                                  valid_sets=[valid_data],
                                  verbose_eval=False,
                                  callbacks=[self.log.log_evaluation(period=100)],
                                  )

            model_name = str(
                self.params.model_dir_path / f'lgb_model_{store_id}_{self.params.prediction_horizon}.bin')
            feature_importance_store_df = pd.DataFrame(sorted(zip(enable_features, estimator.feature_importance())),
                                                       columns=['feature_name', 'importance'])
            feature_importance_store_df = feature_importance_store_df.sort_values('importance', ascending=False)
            feature_importance_store_df['store_id'] = store_id
            feature_importance_store_df.to_csv(
                self.params.result_dir_path / ('feature_importance_{0}_{1}.csv'.format(
                    store_id, self.params.prediction_horizon)), index=False)
            feature_importance_all_df = pd.concat([feature_importance_all_df, feature_importance_store_df])
            pickle.dump(estimator, open(model_name, 'wb'))

            del train_data, valid_data, estimator
            gc.collect()

        self.log.info('aggregate feature importance')
        feature_importance_all_df.to_csv(self.params.result_dir_path / 'feature_importance_all_{0}.csv'.format(
            self.params.prediction_horizon), index=False)
        feature_importance_agg_df = feature_importance_all_df.groupby(
            'feature_name')['importance'].agg(['mean', 'std']).reset_index()
        feature_importance_agg_df.columns = ['feature_name', 'importance_mean', 'importance_std']
        feature_importance_agg_df = feature_importance_agg_df.sort_values('importance_mean', ascending=False)
        feature_importance_agg_df.to_csv(self.params.result_dir_path / 'feature_importance_agg_{0}.csv'.format(
            self.params.prediction_horizon), index=False)

        self.log.info('load base_test')
        base_test = self.load_base_test(store_id_set_list)

        if self.params.export_all_flag:
            base_test.to_csv(
                self.params.result_dir_path / 'exp_base_test_{0}_a.csv'.format(self.params.prediction_horizon),
                index=False)
        if self.params.prediction_horizon_prev > 0:
            pred_v_prev_df = None
            for ph in self.params.prediction_horizon_list:
                if ph <= self.params.prediction_horizon_prev:
                    pred_v_temp_df = pd.read_csv(self.params.result_dir_path / 'pred_v_{}.csv'.format(ph))
                    pred_v_prev_df = pd.concat([pred_v_prev_df, pred_v_temp_df])
            for predict_day in range(1, self.params.prediction_horizon_prev + 1):
                base_test[self.params.target][base_test['d'] == (self.params.end_train_day_x + predict_day)] = \
                    pred_v_prev_df[self.params.target][
                        pred_v_prev_df['d'] == (self.params.end_train_day_x + predict_day)].values

        if self.params.export_all_flag:
            base_test.to_csv(
                self.params.result_dir_path / 'exp_base_test_{0}_b.csv'.format(self.params.prediction_horizon),
                index=False)

        main_time = time.time()
        pred_h_df = pd.DataFrame()
        for predict_day in range(self.params.prediction_horizon_prev + 1, self.params.prediction_horizon + 1):
            self.log.info('predict day{:02d}'.format(predict_day))
            start_time = time.time()
            grid_df = base_test.copy()

            if self.params.recursive_feature_flag:
                self.log.info('[{0} - {1}] calculate recursive features'.format(
                    self.params.end_train_day_x, self.params.prediction_horizon))
                for num_shift_rolling_day in self.params.num_shift_rolling_day_list:
                    num_shift_day = num_shift_rolling_day[0]
                    num_rolling_day = num_shift_rolling_day[1]
                    lag_df = base_test[['id', 'd', self.params.target]]
                    col_name = 'rolling_mean_tmp_' + str(num_shift_day) + '_' + str(num_rolling_day)
                    lag_df[col_name] = lag_df.groupby(['id'])[self.params.target].transform(
                        lambda x: x.shift(num_shift_day).rolling(num_rolling_day).mean())
                    grid_df = pd.concat([grid_df, lag_df[[col_name]]], axis=1)

            day_mask = base_test['d'] == (self.params.end_train_day_x + predict_day)
            if self.params.export_all_flag:
                self.log.info('export recursive_features')
                grid_df[day_mask].to_csv(self.params.result_dir_path / 'exp_recursive_features_{0}_{1}.csv'.format(
                    self.params.prediction_horizon, predict_day), index=False)
            for store_index, store_id in enumerate(store_id_set_list):
                self.log.info('[{3} - {4}] predict {0}/{1} {2} day {5}'.format(
                    store_index + 1, len(store_id_set_list), store_id,
                    self.params.end_train_day_x, self.params.prediction_horizon, predict_day))

                model_path = str(
                    self.params.model_dir_path / f'lgb_model_{store_id}_{self.params.prediction_horizon}.bin')

                estimator = pickle.load(open(model_path, 'rb'))
                if store_id != 'all':
                    store_mask = base_test['store_id'] == store_id
                    mask = (day_mask) & (store_mask)
                else:
                    mask = day_mask

                if self.params.export_all_flag:
                    self.log.info('export pred')
                    grid_df[mask].to_csv(
                        self.params.result_dir_path / (
                                'exp_pred_' + store_id + '_day_' + str(predict_day) + '.csv'), index=False)
                base_test[self.params.target][mask] = estimator.predict(grid_df[mask][self.params.enable_features])

            temp_df = base_test[day_mask][['id', self.params.target]]
            temp_df.columns = ['id', 'F' + str(predict_day)]
            if 'id' in list(pred_h_df):
                pred_h_df = pred_h_df.merge(temp_df, on=['id'], how='left')
            else:
                pred_h_df = temp_df.copy()

            del temp_df

        if self.params.export_all_flag:
            base_test.to_csv(
                self.params.result_dir_path / 'exp_base_test_{0}_c.csv'.format(self.params.prediction_horizon),
                index=False)
        pred_h_df.to_csv(self.params.result_dir_path / 'pred_h_{}.csv'.format(
            self.params.prediction_horizon), index=False)

        pred_v_df = base_test[
            (base_test['d'] >= self.params.end_train_day_x + self.params.prediction_horizon_prev + 1) *
            (base_test['d'] < self.params.end_train_day_x + self.params.prediction_horizon + 1)
            ][
            self.params.main_index_list + [self.params.target]
            ]
        pred_v_df.to_csv(self.params.result_dir_path / 'pred_v_{}.csv'.format(self.params.prediction_horizon),
                         index=False)

        return pred_h_df, pred_v_df

    def calc_wrmsse(self, train_df, prices_df, calendar_df, submission_df, all_preds):
        self.log.info('calc wrmsse')
        temp_df = train_df
        self.log.info('adjust end of train period')
        num_before = train_df.shape
        num_diff_days = self.params.end_train_day_default - self.params.end_train_day_x - \
                        self.params.prediction_horizon
        if num_diff_days > 0:
            temp_df = train_df.iloc[:, :-1 * num_diff_days]
        num_after = temp_df.shape
        self.log.info(num_before, '-->', num_after)

        train_fold_df = temp_df.iloc[:, :-28]
        valid_fold_df = temp_df.iloc[:, -28:].copy()

        valid_preds = submission_df[submission_df['id'].str.contains('evaluation')][['id']]
        valid_preds = valid_preds.merge(all_preds, on=['id'], how='left').fillna(0)
        valid_preds = valid_preds.drop('id', axis=1)
        valid_preds.columns = valid_fold_df.columns
        train_fold_df.to_csv(self.params.result_dir_path / 'eval_wrmsse_train.csv', index=False)
        valid_fold_df.to_csv(self.params.result_dir_path / 'eval_wrmsse_test.csv', index=False)
        valid_preds.to_csv(self.params.result_dir_path / 'eval_wrmsse_pred.csv', index=False)

        evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar_df, prices_df)
        wrmsse = evaluator.score(valid_preds)
        self.log.info('wrmsse', wrmsse)

        return wrmsse

    def main(self):
        self.log.info('main')
        train_df, prices_df, calendar_df, submission_df = self.load_data()

        result_dir_org_path = self.params.result_dir_path
        work_dir_org_path = self.params.work_dir_path
        model_dir_org_path = self.params.model_dir_path
        result_summary_all_df = pd.DataFrame()
        for end_train_day_x in self.params.end_train_day_x_list:
            self.params.end_train_day_x = end_train_day_x
            self.params.result_dir_path = result_dir_org_path / str(end_train_day_x)
            self.params.work_dir_path = work_dir_org_path / str(end_train_day_x)
            self.params.model_dir_path = model_dir_org_path / str(end_train_day_x)
            self.params.update_file_path()

            pred_h_all_df = pd.DataFrame()
            pred_v_all_df = pd.DataFrame()
            self.params.prediction_horizon_prev = 0
            for predict_horizon in self.params.prediction_horizon_list:
                self.log.info('-----------------', 'fold_id', end_train_day_x, 'predict_horizon', predict_horizon)
                self.params.prediction_horizon = predict_horizon
                self.params.update_predict_horizon()
                self.generate_grid_full(train_df, prices_df, calendar_df)
                self.generate_lag_feature()
                self.generate_target_encoding_feature()
                pred_h_df, pred_v_df = self.train_and_predict(train_df, calendar_df, prices_df, submission_df)
                if pred_h_all_df.shape[1] == 0:
                    pred_h_all_df = pred_h_df
                else:
                    pred_h_all_df = pred_h_all_df.merge(pred_h_df, on='id')
                pred_v_all_df = pd.concat([pred_v_all_df, pred_v_df], axis=0)
                self.params.prediction_horizon_prev = predict_horizon

                try:
                    self.log.info('clear work_dir')
                    shutil.rmtree(self.params.work_dir_path)
                    os.mkdir(self.params.work_dir_path)
                except Exception:
                    self.log.exception()

            pred_h_all_df.to_csv(self.params.result_dir_path / 'pred_h_all.csv', index=False)
            pred_v_all_df.to_csv(self.params.result_dir_path / 'pred_v_all.csv', index=False)

            holdout_df = pd.read_csv(self.params.holdout_path)
            self.log.info('holdout_df.shape', holdout_df.shape)
            self.log.info('pred_v_all_df.shape', pred_v_all_df.shape)

            if holdout_df.shape[0] == 0:
                self.log.info('no holdout')
                self.log.info('generate submission')
                pred_h_all_df = pred_h_all_df.reset_index(drop=True)
                submission = pd.read_csv(self.params.raw_submission_path)[['id']]
                submission = submission.merge(pred_h_all_df, on=['id'], how='left').fillna(0)
                submission.to_csv(self.params.result_submission_path, index=False)
                result_summary_df = None
            else:
                self.log.info('calc metrics')
                result_df = holdout_df.merge(pred_v_all_df, on=['id', 'd'], how='inner')
                result_df.columns = ['id', 'd', 'y_test', 'y_pred']
                self.log.info('result_df.shape', pred_v_all_df.shape)
                result_df.to_csv(self.params.result_dir_path / 'result.csv', index=False)

                wrmsse = self.calc_wrmsse(train_df, prices_df, calendar_df, submission_df, pred_h_all_df)

                rmse = np.sqrt(mean_squared_error(result_df['y_test'], result_df['y_pred']))

                result_summary_df = pd.DataFrame(
                    [
                        [self.params.end_train_day_x, 'wrmsse', wrmsse],
                        [self.params.end_train_day_x, 'rmse', rmse],
                    ],
                    columns=['fold_id', 'metric_name', 'metric_value'])
                self.log.info(result_summary_df)
                result_summary_df.to_csv(self.params.result_dir_path / 'result_summary.csv', index=False)
                result_summary_all_df = pd.concat([result_summary_all_df, result_summary_df])

        if result_summary_all_df.shape[0] == 0:
            pass
        else:
            self.log.info(result_summary_all_df)
            self.log.info(result_summary_all_df.groupby('metric_name')['metric_value'].agg(['mean', 'median']))
            result_summary_all_df.to_csv(result_dir_org_path / 'result_summary_all.csv', index=False)
        self.params.reset_dir_path()
        try:
            self.log.info('clear work_dir')
            shutil.rmtree(self.params.work_dir_path)
        except Exception:
            self.log.exception()
        return


if __name__ == '__main__':
    from m5a_params import *

    m = M5aMain(params)
    m.log.info('******** start')
    m.log.info(parser.parse_args())
    m.start_dt = datetime.datetime.now()

    m.log.info(m.params.setting)
    m.main()
    m.log.info(
        ['******** end', 'start_time', m.start_dt, 'process_time', datetime.datetime.now() - m.start_dt])
