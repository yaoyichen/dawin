
#%%
import pandas as pd
import os
import numpy as np
import pickle
import pandas_ta as ta
import matplotlib.pyplot as plt

import xgboost as xgb
import scipy


data_dir = "../../data/baostock_daily/"


file_list = os.listdir(data_dir)

df_all = pd.DataFrame()
"""
暂时看生成一遍特征的时间为
"""
for index, file in enumerate(file_list[0:1000]):
    print(file)
    if(not file.endswith(".csv")):
        continue
        
    df = pd.read_csv(os.path.join(data_dir, file))
    if(len(df) <= 60):
        continue
    df["stock_code"] = file.split(".")[0].split("_")[1]
    df["increase"] = df['close']/df['open']
    df_next1 = df.shift(periods= -1)


    df['next1_open'] = df_next1['open']
    df['next1_close'] = df_next1['close']
    df['label'] = df['next1_open']/df['open'] - 1
    df['label2'] = df['next1_close']/df['close'] - 1



    df_pre1 = df.shift(periods = 1)
    df['p1_increase'] = df_pre1['increase']
    
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    df['Average'] = df.ta.midprice(length=1) #midprice
    df['MA5'] = df.ta.sma(length=5)
    df['MA15'] = df.ta.sma(length=15)
    df['MA60'] = df.ta.sma(length=60)
                  
    
    if(index == 0):
        df_all = df
    else:
        df_all = pd.concat([df_all, df])
        
    

#%%
df_all["MA15_ratio"]  =  (df_all['close'] - df_all['MA15'])/df_all['close']
df_all["MA60_ratio"]  =  (df_all['close'] - df_all['MA60'])/df_all['close']


df_train = df_all.dropna(subset = ["MA60",'label'])
df_train = df_train[(df_train['date'] >= "2017-01-01")  &  (df_train['date'] < "2022-03-01")]

df_test = df_all.dropna(subset = ["MA60",'label'])
df_test = df_test[(df_test['date'] >= "2022-03-01") &  (df_test['date'] <= "2022-06-02")]

# select_features = ["ATR","RSI"]
select_features = ["ATR","RSI","MA60_ratio","MA15_ratio"]
A_train = df_train[['date','stock_code','open']]
X_train = df_train[select_features]
y_trian = df_train[['label']]

A_test = df_test[['date','stock_code','open']]
X_test = df_test[select_features]
y_test = df_test[['label']]

A_test.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)


regressor = xgb.XGBRFRegressor(max_depth = 5)
regressor.fit(X_train,y_trian )

# %%

y_pred = regressor.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns =[ "predict"]) 
predict_result_merge= pd.concat([A_test,X_test,y_test,y_pred_df],
                                axis = 1, ignore_index = True )
predict_result_merge.columns = list(A_test.columns) +  list(X_test.columns) + list(y_test.columns) + list(y_pred_df.columns)
plt.scatter(np.asarray(y_test), y_pred, s = 1)

# %%
mean_error = np.mean(np.abs(predict_result_merge["predict"] - predict_result_merge["label"]))


predict_list =  list(predict_result_merge["predict"])
predict_list.sort(reverse = True)
threshold_value = predict_list[int(0.01*len(predict_result_merge)) ]

predict_win_index_002 = predict_result_merge["predict"] > threshold_value
gain_value = np.mean(predict_result_merge[predict_win_index_002]["label"])
win_rate = np.mean(predict_result_merge[predict_win_index_002]["label"] > 0)

print(f"mean error:{mean_error:.4f}")
print(f"threshold:{threshold_value:.4f},gain_value:{gain_value:.4f}, win_rate:{win_rate:.4f}")
plt.scatter(predict_result_merge[predict_win_index_002]["label"],
            predict_result_merge[predict_win_index_002]["predict"], s = 1 )
plt.xlabel("label")
plt.ylabel("predict")
plt.grid()

# %%

# from xgboost import plot_tree
# plot_tree(regressor)
# plt.show()

#%%
tt = predict_result_merge[predict_result_merge["stock_code"] == 
"000831"]
plt.scatter(tt["label"], tt["predict"])
plt.xlabel("label")
plt.ylabel("predict")
plt.grid()

