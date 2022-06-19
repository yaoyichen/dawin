
#%%
import pandas as pd
import os
import numpy as np
import pickle
import pandas_ta as ta
import matplotlib.pyplot as plt

import xgboost as xgb
import scipy


data_dir = "../../data/baostock_daily/20220619_download/"


file_list = os.listdir(data_dir)

data_type = "300"
data_type = "all"
train_folds = 5


def generate_feature(df):
    df["stock_code"] = file.split(".")[0].split("_")[1]
    df["increase"] = df['close']/df['open']
    
    df['volume_MA3'] = ta.sma(df["volume"], length = 3)
    df['volume_MA5'] = ta.sma(df["volume"], length = 5)
    
    df['MA5'] =  ta.sma(df["close"], length = 5)
    df['MA10'] = ta.sma(df["close"], length = 10)
    df['MA20'] = ta.sma(df["close"], length = 20)
    df['MA30'] = ta.sma(df["close"], length = 30)
    df['MA60'] = ta.sma(df["close"], length = 60)
    
    
    df['turn_MA3'] =  ta.sma(df["turn"], length = 3)
    df['turn_MA5'] =  ta.sma(df["turn"], length = 5)
    df['turn_MA10'] = ta.sma(df["turn"], length = 10)
    df['turn_MA20'] = ta.sma(df["turn"], length = 20)
    df['turn_MA60'] = ta.sma(df["turn"], length = 60)

    
    # df[["KVO", "KVOs"]] =  df.ta.kvo(fast=34, slow=55, signal=13, drift=1) 
    
    df_pre1 = df.shift(periods = 1)

    
    df[["DCL", "DCM", "DCU"]] = df.ta.donchian(lower_length=14, upper_length=14)
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    
    
    # volume
    """
    Accumulation/Distribution Index: ad
    Accumulation/Distribution Oscillator: adosc
    Archer On-Balance Volume: aobv
    Chaikin Money Flow: cmf
    Elder's Force Index: efi
    Ease of Movement: eom
    Klinger Volume Oscillator: kvo
    Money Flow Index: mfi
    Negative Volume Index: nvi
    On-Balance Volume: obv
    Positive Volume Index: pvi
    Price-Volume: pvol
    Price Volume Rank: pvr
    Price Volume Trend: pvt
    Volume Profile: vp
    """
    df['AD'] = df.ta.ad()
    df['ADOSC'] = df.ta.adosc()
    df[['OBV', 'OBV_min_2', 'OBV_max_2', 'OBVe_4', 'OBVe_12', 'AOBV_LR_2',
           'AOBV_SR_2']] = df.ta.aobv()
    df["CMF"] =  df.ta.cmf()
    df["EOM"] = df.ta.eom()
    # df[["KVO","KVO2"]] = df.ta.kvo()
    df["MFI"] = df.ta.mfi()
    df["NVI"] = df.ta.nvi()
    
    df["obv"] = df.ta.obv()
    df['PVI'] = df.ta.pvi()
    df['PVOL'] = df.ta.pvol()
    df['PVR'] = df.ta.pvr()
    # df['PVT'] = df.ta.pvt()
    # df[["low_close",'mean_close','high_close','pos_volume', 'neg_volume', "total_volume"]] = df.ta.vp()
    
    
    df[['bbl','bbm','bbu','bbb','bbp']] = df.ta.bbands(length = 20)
    df['Average'] = df.ta.midprice(length=1) #midprice
    df["tail_up"] = df["close"]/ df['Average'] - 1 
    df[["macd_","macd_f","macd_s"]] = df.ta.macd(fast = 9, slow = 31)
    

    
    df["volume_MA5_increase"] = df['volume_MA5']/df_pre1['volume_MA5']
    df["MA5_increase"] = df['MA5']/df_pre1['MA5']
       
    
    return df

result_list = []
for i in range(train_folds):
    df_all = pd.DataFrame()
    print(f"part:{i}")    
    """
    暂时看生成一遍特征的时间为
    """
    # for index, file in enumerate(file_list[0:5000:200]):
    for index, file in enumerate(file_list[i:5000:train_folds]):
        
        # print(file)
        
        if(not file.endswith(".csv")):
            continue
        stock_code = file.split("_")[1].split(".")[0]
        
        if(data_type =="300"):
            if(not(( stock_code >= "300000")  and (stock_code < "399999") ) ):
                continue
        
        df = pd.read_csv(os.path.join(data_dir, file))

        
        
        # df = df[(df["date"]> "2021-12-01") & (df["date"]<= "2022-06-12") ]
        df = df[(df["date"]> "2019-12-01") ]
        if(len(df) <= 60):
            continue
        
                
        """
        排除掉st股
        """
        if(df[["isST"]].values[-1][0] == 1):
            continue
        
        df = generate_feature(df)                                    
    
        # 这个只有取label的时候才被用到
        df_next1 = df.shift(periods= -1)
        df['next1_increase'] = df_next1['increase']
        
        df_next3 = df.shift(periods= -3)
        df['next3_increase'] = (df_next3['close'])/df_next1["open"]
        
        
        df_next5 = df.shift(periods= -5)
        df['next5_increase'] = (df_next5['close'])/df_next1["open"]
        
        
        df_next10 = df.shift(periods= -10)
        df['next10_increase'] = (df_next10['close'])/df_next1["open"]
        
        
        
        if(index == 0):
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
            
        
    
    #%%
    def second_fine(df_all):
        df_all["MA5_ratio"]  =  (df_all['close'] - df_all['MA5'])/df_all['close']
        df_all["MA10_ratio"]  =  (df_all['close'] - df_all['MA10'])/df_all['close']
        df_all["MA20_ratio"]  =  (df_all['close'] - df_all['MA20'])/df_all['close']
        df_all["MA30_ratio"]  =  (df_all['close'] - df_all['MA30'])/df_all['close']
        df_all["MA60_ratio"]  =  (df_all['close'] - df_all['MA60'])/df_all['close']
        
        
        df_all["turn_MA3_ratio"]  =  (df_all['turn'] - df_all['turn_MA3'])/df_all['turn']
        df_all["turn_MA5_ratio"]  =  (df_all['turn'] - df_all['turn_MA5'])/df_all['turn']
        df_all["turn_MA10_ratio"]  =  (df_all['turn'] - df_all['turn_MA10'])/df_all['turn']
        df_all["turn_MA20_ratio"]  =  (df_all['turn'] - df_all['turn_MA20'])/df_all['turn']
        df_all["turn_MA60_ratio"]  =  (df_all['turn'] - df_all['turn_MA60'])/df_all['turn']
        
        df_all["turn_MA3_5_ratio"]  =  (df_all['turn_MA3'] - df_all['turn_MA5'])/df_all['turn_MA3']
        df_all["turn_MA3_10_ratio"]  =  (df_all['turn_MA3'] - df_all['turn_MA10'])/df_all['turn_MA3']
        
        
        # df_all["DCU_uppper"] = df['close'] > df_all["DCU"]
        # df_all["DCU_lower"] = df['close'] - df_all["DCU"] 
        
        
        """
        相比于前几天的量比增加量
        """
        df_all["volume_MA3_ratio"]  =  (df_all['volume'] - df_all['volume_MA3'])/df_all['volume_MA3']
        df_all["volume_MA5_ratio"]  =  (df_all['volume'] - df_all['volume_MA5'])/df_all['volume_MA5']
        
        # 去除成交量为0
        df_all = df_all[df_all['volume'] !=0]
        df_all = df_all[df_all['volume_MA5'] >0 ]
        df_all = df_all[df_all['volume_MA3'] >0 ]
        df_all = df_all[df_all['turn_MA3'] >0]
        df_all = df_all[df_all['turn'] > 0]
        
        df_all = df_all[df_all["volume_MA5_increase"] < 10]
        df_all = df_all[df_all["MA5_increase"] < 10]
        return df_all
    
    df_all = second_fine(df_all)
    
    
    #%%
    select_label = ["next3_increase"]  
    
    turn_features =  ["turn", 'turn_MA3',"turn_MA3_ratio","turn_MA5_ratio","turn_MA10_ratio","turn_MA20_ratio",
                      "turn_MA60_ratio","turn_MA3_5_ratio","turn_MA3_10_ratio"]
        
    macd_features = ["macd_", "macd_f", "macd_s"]
    volume_features = ["EOM", "CMF"]
    
    select_features = ["MA5_ratio", "MA10_ratio", "MA20_ratio", "MA30_ratio",'bbb','bbp',"tail_up","ATR","RSI"] \
        + turn_features + macd_features + volume_features
    # + volume_features
    
    
    
    df_train = df_all.dropna(subset = ["MA60",select_label[0],"volume_MA5_increase","MA5_increase"])
    df_train = df_train.dropna(subset = select_features)
    # df_train = df_train[(df_train['date'] >= "2019-01-01")  &  (df_train['date'] <= "2022-02-01")]
    df_train = df_train[(df_train['date'] >= "2020-01-01") ]
       
    
    df_test = df_all.dropna(subset = ["MA60",select_label[0],"volume_MA5_increase","MA5_increase"])
    df_test = df_test.dropna(subset = select_features)
    # df_test = df_test[(df_test['date'] >= "2022-03-01") &  (df_test['date'] <= "2022-06-02")]
    df_test = df_test[(df_test['date'] >= "2022-05-01")]
    
    
    # df_train = df_train[(df_train["stock_code"] >= "300000") & (df_train["stock_code"] <= "400000")]
    # df_test = df_test[(df_test["stock_code"] >= "300000") & (df_test["stock_code"] <= "400000")]
   
    
    # df_train = df_train[(df_train["stock_code"] >= "680000") & (df_train["stock_code"] <= "800000")]
    # df_test = df_test[(df_test["stock_code"] >= "680000") & (df_test["stock_code"] <= "800000")]
   
    
   
    # select_features = ["ATR","RSI"]
    # select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","volume_MA5_ratio"]
    
    # select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","volume_MA5_ratio",
    #                    "volume_MA5_increase","MA5_increase","tail_up","increase",
    #                    'bbl','bbm','bbu','bbb','bbp',"macd_", "macd_f", "macd_s"]
    
    volume_features = ['AD','ADOSC','OBV', 'OBV_min_2', 'OBV_max_2', 'OBVe_4', 'OBVe_12', 'AOBV_LR_2',
    'AOBV_SR_2',"CMF","EOM","MFI","NVI","obv",'PVI','PVOL','PVR','PVT']


    # select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","MA20_ratio","MA30_ratio","volume_MA5_ratio",
    #                     "volume_MA5_increase","MA5_increase","tail_up","increase",
    #                     'bbl','bbm','bbu','bbb','bbp',"macd_", "macd_f", "macd_s","DCL", "DCM","DCU"] 

    
    
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
    
    predict_result_merge = predict_result_merge[ predict_result_merge[select_label[0]]> -100]
    plt.scatter(np.asarray(y_test), y_pred, s = 1)
    
    
    plt.figure(1)
    result_gp  = predict_result_merge.groupby(["date"]).mean()
    plt.plot((result_gp["predict"] -1) *10,"-x",label = "predict")
    plt.plot(result_gp[select_label[0]] -1,"-x", label = "true")
    plt.legend()
    
    # %%
    mean_error = np.mean(np.abs(predict_result_merge["predict"] - predict_result_merge[select_label[0]]))
    
    predict_list =  list(predict_result_merge["predict"])
    predict_list.sort(reverse = True)
    threshold_value_upper = predict_list[int(0.000*len(predict_result_merge)) ]
    
    # threshold_value_upper = 1.02
    threshold_value_lower = predict_list[int(0.005*len(predict_result_merge)) ]
    # threshold_value_lower = -1
    
    predict_win_index_002 = (predict_result_merge["predict"] <= threshold_value_upper) & (predict_result_merge["predict"] > threshold_value_lower)
    
    fetch_pd = predict_result_merge[predict_win_index_002]
    
    gain_value = np.mean(fetch_pd[select_label[0]]) - 1
    win_rate = np.mean(fetch_pd[select_label[0]] > 1.0)
    
    print(f"mean error:{mean_error:.4f}, sample_number:{np.sum(predict_win_index_002 == True)}")
    print(f"threshold:{threshold_value_upper:.4f},gain_value:{gain_value:.4f}, win_rate:{win_rate:.4f}")
    print(f"base line all:{predict_result_merge[select_label[0]].mean()}")
    print("day average:{}" + str(fetch_pd.groupby(["date"]).mean()[ select_label[0]].mean() ) )
    plt.scatter(predict_result_merge[predict_win_index_002][select_label[0]],
                predict_result_merge[predict_win_index_002]["predict"], s = 1 )
    
    plt.xlabel(select_label[0])
    plt.ylabel("predict")
    plt.grid()
    
    #%%
    tt = predict_result_merge[predict_result_merge["date"] == "2022-03-01"]
    predict_list =  list(tt["predict"])
    predict_list.sort(reverse = True)
    # threshold_value_upper = predict_list[10]
    # predict_win_index_002 = 
    
    
    
    
    # %%
    
    # from xgboost import plot_tree
    # plot_tree(regressor)
    # plt.show()
    
    plt.figure( figsize= ( 3,10) )
    feature_importances = regressor.feature_importances_ 
    # select_features = ["ATR","RSI","MA60_ratio","MA15_ratio","volume_MA5_ratio",
    #                    "volume_MA5_increase","MA5_increase","tail_up","increase",
    #                    'bbl','bbm','bbu','bbb','bbp',"macd_", "macd_f", "macd_s"]
    
    feature_importances_argsort = feature_importances.argsort()
    
    plt.barh([select_features[i] for i in feature_importances_argsort],
            feature_importances[feature_importances_argsort] )
    
    # plot pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_) pyplot.show() 作者：python风控模型 https://www.bilibili.com/read/cv12836121/ 出处：bilibili
    
    
    
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
    data_dir = "../../data/baostock_daily/20220619_download"
    
    first_data = 1
    for index, file in enumerate(file_list[0:5000:1]):
        # print(file)
        if(not file.endswith(".csv")):
            continue
        stock_code = file.split("_")[1].split(".")[0]
        
        df = pd.read_csv(os.path.join(data_dir, file))
        if(len(df) <= 60):
            continue
        
        # print("000")
        if(data_type =="300"):
            if(not(( stock_code >= "300000")  and (stock_code < "399999") ) ):
                continue
        # print("100")
        """
        排除掉st股
        """
        if(df[["isST"]].values[-1][0] == 1):
            continue
        
        df_today = df[-60::]
        df_today = df_today.reset_index(drop = True )
        df_today_feature = generate_feature(df_today)    
        
        
        if(first_data == 1):
            df_all_today = df_today_feature.loc[[59]]
            first_data = 0
            # print("0")
        else:
            df_all_today = pd.concat([df_all_today, df_today_feature.loc[[59]] ])
            # print("1")
            
    df_all_today_refine = second_fine(df_all_today)
        
    
    A_inference = df_all_today_refine[['date','stock_code','open']]
    X_inference = df_all_today_refine[select_features]
    
    A_inference.reset_index(drop = True, inplace = True)
    A_inference.reset_index(drop = True, inplace = True)
    
    y_pred = regressor.predict(X_inference)
    y_pred_df = pd.DataFrame(y_pred, columns =[ "predict"]) 
    predict_result_merge = pd.concat([A_inference,X_inference,y_pred_df],
                                axis = 1, ignore_index = True )
    predict_result_merge.columns = list(A_inference.columns) +  list(X_inference.columns) + list(y_pred_df.columns)
    
    #%%
    result_list.append(predict_result_merge)

# predict_win_index_002 = predict_result_merge["predict"] > 1.032

# print(predict_result_merge[predict_win_index_002])


# stock_zh_a_spot_em_df, now, query_elapse_time  = get_current_state()

# result = pd.merge([predict_win_index_002, stock_zh_a_spot_em_df],
#                   left_on= [], right_on= [], how = "inner")

#%% 选择2个，各25%仓入, 月底要 6%收益。 
# df_result1 = result_list[0]
# df_result2 = result_list[1]
# df_result3 = result_list[2]
# df_result4 = result_list[3]
# df_result5 = result_list[4] 

"""
threshold:1.0493,gain_value:0.0728, win_rate:0.8495
threshold:1.0535,gain_value:0.0696, win_rate:0.8379
threshold:1.0452,gain_value:0.0705, win_rate:0.8397
threshold:1.0321,gain_value:0.0638, win_rate:0.7556
threshold:1.0580,gain_value:0.0665, win_rate:0.7792

"""

#%%
"""
获得merge表， 
"""
import os
os.sys.path.append("..")
from tools.download_tools import *
df_final = pd.merge(df_result1[["stock_code","predict"]] ,
                    df_result2[["stock_code","predict"]] , 
                   left_on= ["stock_code"], right_on= ["stock_code"], how = "inner" )


df_final2 = pd.merge(df_final[["stock_code","predict_x","predict_y"]] ,
                    df_result3[["stock_code","predict"]] , 
                   left_on= ["stock_code"], right_on= ["stock_code"], how = "inner" )

df_final2= df_final2.rename(columns = {"predict":"predict_z"})

df_final3 = pd.merge(df_final2[["stock_code","predict_x","predict_y","predict_z"]] ,
                    df_result4[["stock_code","predict"]] , 
                   left_on= ["stock_code"], right_on= ["stock_code"], how = "inner" )

df_final3= df_final3.rename(columns = {"predict":"predict_a"})


df_final = pd.merge(df_final3[["stock_code","predict_x","predict_y","predict_z","predict_a"]] ,
                    df_result5[["stock_code","predict"]] , 
                   left_on= ["stock_code"], right_on= ["stock_code"], how = "inner" )
# 
# ["stock_code","predict_x","predict_y","predict_z","predict_a","predict"]
#%%


def merge_prediction_result(df_final,keep_columns = ["stock_code","predict"]):
    stock_zh_a_spot_em_df, now, query_elapse_time  = get_current_state()
    stock_zh_a_spot_em_df = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["最新价"] > 0]
    stock_zh_a_spot_em_df = stock_zh_a_spot_em_df[~stock_zh_a_spot_em_df["名称"].str.contains("退")]
    df_final_2022 = pd.merge(df_final[keep_columns] ,
                        stock_zh_a_spot_em_df[["代码","名称"]] , 
                        left_on= ["stock_code"], right_on= ["代码"], how = "inner" )
    # df_final_2022["predict_mean"] = (df_final_2022["predict_x"] + df_final_2022["predict_x"] +  df_final_2022["predict_z"] +  \
    #     df_final_2022["predict_a"] + df_final_2022["predict"])/5.0
    return df_final_2022
    
result = merge_prediction_result(df_final, keep_columns = ["stock_code","predict"])
df_final_2022.to_csv("20220619_3.csv", index = False)
    
#%%


threshold = 1.01
tt = df_final_2022[(df_final_2022["predict_x"] > threshold) & 
              (df_final_2022["predict_y"] > threshold) &
              (df_final_2022["predict_z"] > threshold) & 
                (df_final_2022["predict_a"] > threshold) &
                (df_final_2022["predict"] > threshold)  
              ]

print(list(tt["名称"]))


result_sort = df_final_2022.sort_values(by = "predict_mean", ascending = False)
print(list(result_sort[0:20]["名称"]))


hand_list1 = ["东方财富","宁波精达","景嘉微","创识科技","四方精创","天阳科技","奥翔药业","美诺华"]
# hand_list2 = ["创维数字","申通快递",]
tt = result_sort[result_sort["名称"].isin(hand_list1) ]

print(tt)

#%%

result_sort = result.sort_values(by = "predict", ascending = False)
print(list(result_sort[0:20]["名称"]))

