
#%%
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy
# %%
data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(data, label=label)
# %%
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]


num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)
# %%



data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)
# %%

xgb.plot_tree(bst, num_trees=2)
# %%
