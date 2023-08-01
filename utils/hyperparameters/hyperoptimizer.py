import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, STATUS_OK


class HpOpt(object):
    def __init__(self, X_train, X_test, y_train, y_test, loss_func, random_state, max_evals, feature_name, best_trial_dir_path, model_dir_path):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.loss_func = loss_func
        self.random_state = random_state
        self.max_evals = max_evals
        self.feature_name = feature_name

        self.best_trial_dir_path = best_trial_dir_path
        self.model_dir_path = model_dir_path

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            best = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            return best, trials
        except Exception as e:
            print(e)
    
    def store_best_trial(self, best_trail, file_name, feature_name, compare=True):
        best_trail = {self.random_state: best_trail}

        if not os.path.exists(f'{self.best_trial_dir_path}'):
            os.makedirs(f'{self.best_trial_dir_path}')

        best_trail_path = f'{self.best_trial_dir_path}/{file_name}.{feature_name}.pkl'
        if compare:
            # compare with previous best trial
            if os.path.exists(best_trail_path):
                pre_best_trial = pickle.load(open(best_trail_path, 'rb'))
                if self.random_state in pre_best_trial.keys():
                    if best_trail[self.random_state]['result']['loss'] < pre_best_trial[self.random_state]['result']['loss']:
                        print('Update best trial')
                        pre_best_trial.update(best_trail)
                        pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
                        return
                    else:
                        print('No update best trial')
                        pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
                else:
                    print('Add best trial')
                    pre_best_trial.update(best_trail)
                    pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
            else:
                print('Initialize best trial')
                pickle.dump(best_trail, open(best_trail_path, 'wb'))

        else:
            if os.path.exists(best_trail_path):
                print('Update best trial')
                pre_best_trial = pickle.load(open(best_trail_path, 'rb'))
                pre_best_trial.update(best_trail)
                pickle.dump(pre_best_trial, open(best_trail_path, 'wb'))
            else:
                print('Create best trial')
                pickle.dump(best_trail, open(best_trail_path, 'wb'))

    def rf_cli(self, params):
        # RandomForestClassifier
        cli = RandomForestClassifier(**params)
        fit_params = {}
        return self.train_cli(cli, fit_params)
    
    def xgb_cli(self, params):
        # XGBClassifier gpu version
        cli = XGBClassifier(gpu_id=0, tree_method='gpu_hist', eval_metric='mlogloss',
                            early_stopping_rounds=5, **params) 
        fit_params = {'eval_set': [(self.X_test, self.y_test)], 'verbose': False}
        return self.train_cli(cli, fit_params)
    
    def train_cli(self, cli, fit_params):
        cli.fit(self.X_train, self.y_train, **fit_params)
        pred = cli.predict_proba(self.X_test)
        loss = self.loss_func(self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
    