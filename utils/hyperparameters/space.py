import numpy as np
import torch
from hyperopt import hp


class HpOptParametersSpace():
    '''
    Hyperopt is not fully support nested space.
    Not recommended to aggregate all parameters for certain model into one space.
    '''
    
    def __init__(self):
        # logistic regression
        self.lr_cli_params = {
            'C':                        hp.loguniform('C', -5, 0),
            'penalty':                  hp.choice('penalty', ['l1', 'l2']),
            'solver':                   hp.choice('solver', ['liblinear']),
            'tol':                      hp.loguniform('tol', -4, -2),
            'max_iter':                 hp.choice('max_iter', [1000]),
            'random_state':             hp.choice('random_state', [0]),
        }
    
        # knn
        self.knn_cli_params = {
            'n_neighbors':              hp.choice('n_neighbors', [5, 10, 15, 20, 25, 30]),
            'weights':                  hp.choice('weights', ['uniform', 'distance']),
            'algorithm':                hp.choice('algorithm', ['auto', 'ball_tree']),
            'leaf_size':                hp.choice('leaf_size', [20, 30, 40]),
            'p':                        hp.choice('p', [1, 2]),
        }
    
        # support vector machine
        self.svc_cli_params = {
            'C':                        hp.loguniform('C', -5, 2),
            'kernel':                   hp.choice('kernel', ['poly', 'rbf', 'sigmoid']),  # linear is pretty slow
            'gamma':                    hp.choice('gamma', ['scale', 'auto']),
            'coef0':                    hp.uniform('coef0', 0, 1),
            'tol':                      hp.loguniform('tol', -4, -2),
            'cache_size':               hp.choice('cache_size', [2000]),
            'shrinking':                hp.choice('shrinking', [True, False]),
            'break_ties':               hp.choice('break_ties', [False, True]),
            'class_weight':             hp.choice('class_weight', [None, 'balanced']),
            'probability':              hp.choice('probability', [True]),
        }
    
        # random forest
        self.rf_cli_params = {
            'n_estimators':             hp.choice('n_estimators', np.arange(100, 1001, 100, dtype=int)),
            'criterion':                hp.choice('criterion', ['gini']),
            'max_depth':                hp.choice('max_depth', np.arange(5, 20, dtype=int)),
            'min_samples_split':        hp.choice('min_samples_split', np.arange(2, 11, dtype=int)),
            'min_samples_leaf':         hp.choice('min_samples_leaf', np.arange(1, 11, dtype=int)),
            'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
            'min_impurity_decrease':    hp.uniform('min_impurity_decrease', 0, 0.5),
            'class_weight':             hp.choice('class_weight', [None, 'balanced']),
            'n_jobs':                   hp.choice('n_jobs', [32]),  # manually set
            'max_features':             hp.choice('max_features', ['sqrt']),
        }
    
        # XGB parameters
        xgb_warm_cli_params = {
            'eta':                  hp.uniform('eta', 0.2997, 0.3003),
            'max_depth':            hp.choice('max_depth', [5, 6, 7]),
            'subsample':            hp.uniform('subsample', 0.9999, 1),
            'colsample_bytree':     hp.uniform('colsample_bytree', 0.9999, 1),
            'colsample_bylevel':    hp.uniform('colsample_bylevel', 0.9999, 1),
            'min_child_weight':     hp.uniform('min_child_weight', 0.9999, 1),
            'alpha':                hp.uniform('alpha', 0, 0.0001),
            'lambda':               hp.uniform('lambda', 0.9999, 1),
            'gamma':                hp.uniform('gamma', 0, 0.0001),
        }
    
        self.xgb_cli_params = {
            'eta':                  hp.loguniform('eta', -7, 0),
            'max_depth':            hp.choice('max_depth', np.arange(1, 11, dtype=int)),
            'subsample':            hp.uniform('subsample', 0.2, 1),
            'colsample_bytree':     hp.uniform('colsample_bytree', 0.2, 1),
            'colsample_bylevel':    hp.uniform('colsample_bylevel', 0.2, 1),
            'min_child_weight':     hp.loguniform('min_child_weight', -16, 2),
            'alpha':                hp.uniform('alpha', 0, 1),
            'lambda':               hp.uniform('lambda', 0, 1),
            'gamma':                hp.uniform('gamma', 0, 1),
        }
    
        self.lgb_cli_params = {
            'learning_rate':        hp.loguniform('learning_rate', -5, -2),
            'max_depth':            hp.choice('max_depth', np.arange(3, 11, dtype=int)),
            'num_leaves':           hp.choice('num_leaves', np.arange(8, 51, dtype=int)),
            'min_data_in_leaf':     hp.choice('min_data_in_leaf', np.arange(10, 20, dtype=int)),
            'verbose':              hp.choice('verbose', [-1]),
        }
    
        self.cnn_1d_cli_params = {
            'hidden_size':      hp.choice('hidden_size', [32, 64, 128]),
            'optimizer':        hp.choice('optimizer', [torch.optim.Adam, torch.optim.AdamW]),
            'learning_rate':    hp.loguniform('learning_rate', -5, -2),
            'batch_size':       hp.choice('batch_size', [32, 64, 128, 256, 512])
        }
