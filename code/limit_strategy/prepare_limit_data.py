
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


def generate_feature(df):
    df["stock_code"] = file.split(".")[0].split("_")[1]
    df["increase"] = df['close']/df['open']
    
    df['volume_MA3'] = ta.sma(df["volume"], length = 3)
    df['volume_MA5'] = ta.sma(df["volume"], length = 5)
    
    df['MA5'] =  ta.sma(df["close"], length = 5)
    df['MA15'] = ta.sma(df["close"], length = 15)
    df['MA60'] = ta.sma(df["close"], length = 60)
    
    df_pre1 = df.shift(periods = 1)
    # df['p1_increase'] = df_pre1['increase']
    
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    df[['bbl','bbm','bbu','bbb','bbp']] = df.ta.bbands(length = 20)
    df['Average'] = df.ta.midprice(length=1) #midprice
    df["tail_up"] = df["close"]/ df['Average'] - 1 
    df[["macd_","macd_f","macd_s"]] = df.ta.macd(fast = 9, slow = 31)
    

    
    df["volume_MA5_increase"] = df['volume_MA5']/df_pre1['volume_MA5']
    df["MA5_increase"] = df['MA5']/df_pre1['MA5']
       
    
    return df
    
    
"""
暂时看生成一遍特征的时间为
"""
# for index, file in enumerate(file_list[0:5000:200]):
for index, file in enumerate(file_list[1:5000:10]):
    print(file)
    if(not file.endswith(".csv")):
        continue
        
    df = pd.read_csv(os.path.join(data_dir, file))
    df = df[df["date"]>= "2019-01-01"]
    
    if(len(df) <= 60):
        continue
    df = generate_feature(df)                                    

    # 这个只有取label的时候才被用到
    df_next1 = df.shift(periods= -1)
    df['next1_increase'] = df_next1['increase']
    
    df_next3 = df.shift(periods= -3)
    df['next3_increase'] = (df_next3['close'])/df_next1["open"]
    
    
    
    if(index == 0):
        df_all = df
    else:
        df_all = pd.concat([df_all, df])
        
    

#%%
def second_fine(df_all):
    df_all["MA5_ratio"]  =  (df_all['close'] - df_all['MA5'])/df_all['close']
    df_all["MA15_ratio"]  =  (df_all['close'] - df_all['MA15'])/df_all['close']
    df_all["MA60_ratio"]  =  (df_all['close'] - df_all['MA60'])/df_all['close']
    
    """
    相比于前几天的量比增加量
    """
    df_all["volume_MA3_ratio"]  =  (df_all['volume'] - df_all['volume_MA3'])/df_all['volume_MA3']
    df_all["volume_MA5_ratio"]  =  (df_all['volume'] - df_all['volume_MA5'])/df_all['volume_MA5']
    
    # 去除成交量为0
    df_all = df_all[df_all['volume'] !=0]
    df_all = df_all[df_all["volume_MA5_increase"] < 10]
    df_all = df_all[df_all["MA5_increase"] < 10]
    
    return df_all

df_all = second_fine(df_all)

select_label = ["next3_increase"]


df_train = df_all.dropna(subset = ["MA60",select_label[0],"volume_MA5_increase","MA5_increase"])
df_train = df_train[(df_train['date'] >= "2017-01-01")  &  (df_train['date'] < "2022-03-01")]

df_test = df_all.dropna(subset = ["MA60",'increase',"volume_MA5_increase","MA5_increase"])
df_test = df_test[(df_test['date'] >= "2022-03-01") &  (df_test['date'] <= "2022-06-02")]

# select_features = ["ATR","RSI"]
select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","volume_MA5_ratio"]

select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","volume_MA5_ratio",
                   "volume_MA5_increase","MA5_increase","tail_up",
                   'bbl','bbm','bbu','bbb','bbp',"macd_", "macd_f", "macd_s"]


A_train = df_train[['date','stock_code','open']]
X_train = df_train[select_features]
y_trian = df_train[select_label]

A_test = df_test[['date','stock_code','open']]
X_test = df_test[select_features]
y_test = df_test[select_label]

A_test.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
#%%

regressor = xgb.XGBRFRegressor(max_depth = 6)
regressor.fit(X_train,y_trian )

# %%
plt.figure(0)
y_pred = regressor.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns =[ "predict"]) 
predict_result_merge= pd.concat([A_test,X_test,y_test,y_pred_df],
                                axis = 1, ignore_index = True )
predict_result_merge.columns = list(A_test.columns) +  list(X_test.columns) + list(y_test.columns) + list(y_pred_df.columns)
plt.scatter(np.asarray(y_test), y_pred, s = 1)


plt.figure(1)
result_gp  = predict_result_merge.groupby(["date"]).mean()
plt.plot((result_gp["predict"] -1) *10,"-x")
plt.plot(result_gp[select_label[0]] -1,"-x")

# %%
mean_error = np.mean(np.abs(predict_result_merge["predict"] - predict_result_merge[select_label[0]]))


predict_list =  list(predict_result_merge["predict"])
predict_list.sort(reverse = True)
threshold_value_upper = predict_list[int(0.005*len(predict_result_merge)) ]

# threshold_value_upper = 1.05
threshold_value_lower = predict_list[int(0.03*len(predict_result_merge)) ]
# threshold_value_lower = -1

predict_win_index_002 = (predict_result_merge["predict"] <= threshold_value_upper) & (predict_result_merge["predict"] > threshold_value_lower)

fetch_pd = predict_result_merge[predict_win_index_002]

gain_value = np.mean(fetch_pd[select_label[0]]) - 1
win_rate = np.mean(fetch_pd[select_label[0]] > 1.0)

print(f"mean error:{mean_error:.4f}, sample_number:{np.sum(predict_win_index_002 == True)}")
print(f"threshold:{threshold_value_upper:.4f},gain_value:{gain_value:.4f}, win_rate:{win_rate:.4f}")
print(fetch_pd.groupby(["date"]).mean()["next3_increase"].mean())
plt.scatter(predict_result_merge[predict_win_index_002][select_label[0]],
            predict_result_merge[predict_win_index_002]["predict"], s = 1 )
plt.xlabel(select_label[0])
plt.ylabel("predict")
plt.grid()

#%%
tt = predict_result_merge[predict_result_merge["date"] == "2022-03-01"]
predict_list =  list(tt["predict"])
predict_list.sort(reverse = True)
threshold_value_upper = predict_list[10]
predict_win_index_002 = 




# %%

# from xgboost import plot_tree
# plot_tree(regressor)
# plt.show()

#%%
# tt = predict_result_merge[predict_result_merge["stock_code"] == 
# "000831"]
# plt.scatter(tt["increase"], tt["predict"])
# plt.xlabel("label")
# plt.ylabel("predict")
# plt.grid()

"""
这份是做inference的代码
"""
data_dir = "../../data/baostock_daily/"

for index, file in enumerate(file_list[0:5000:1]):
    print(file)
    if(not file.endswith(".csv")):
        continue
    
    df = pd.read_csv(os.path.join(data_dir, file))
    if(len(df) <= 60):
        continue
    df_today = df[-60::]
    df_today = df_today.reset_index(drop = True )
    df_today = generate_feature(df_today)    
    
    
    if(index == 0):
        df_all_today = df_today.loc[[59]]
    else:
        
        df_all_today = pd.concat([df_all_today, df_today.loc[[59]] ])
        
        
        
df_all_today_refine = second_fine(df_all_today)
    

A_test = df_all_today_refine[['date','stock_code','open']]
X_test = df_all_today_refine[select_features]

A_test.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)

y_pred = regressor.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns =[ "predict"]) 
predict_result_merge= pd.concat([A_test,X_test,y_pred_df],
                            axis = 1, ignore_index = True )
predict_result_merge.columns = list(A_test.columns) +  list(X_test.columns) +list(y_pred_df.columns)
#%%
predict_win_index_002 = predict_result_merge["predict"] > 1.008
    
print(predict_result_merge[predict_win_index_002])


