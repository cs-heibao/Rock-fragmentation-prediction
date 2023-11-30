"""
XGB岩石强度预测使用贝叶斯优化调参
"""
import xgboost
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, Trials, STATUS_OK, Trials, anneal
from functools import partial
from hyperopt.fmin import fmin
from sklearn.metrics import f1_score,r2_score,mean_squared_error,mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# 特征数据归一化处理
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train_0827.csv")
test_df = pd.read_csv("test1.csv")
train_df.drop('ID', axis=1, inplace=True)
test_df.drop('ID', axis=1, inplace=True)

X_train = train_df.drop(['X50'], axis=1)
y_train = train_df['X50']

X_test = test_df.drop(['X50'], axis=1)
y_test = test_df['X50']

def model_metrics(model, x, y):
    """ 评估指标 """
    yhat = model.predict(x)
    return r2_score(y, yhat)
    # return f1_score(y, yhat, average='micro')


def bayes_fmin(train_x, test_x, train_y, test_y, eval_iters=100):
    """
    bayes优化超参数
    eval_iters：迭代次数
    """

    def factory(params):
        """
        定义优化的目标函数
        """
        fit_params = {
            # 'max_depth': int(params['max_depth']),
            # 'n_estimators': int(params['n_estimators']),
            # 'max_leaf_nodes': int(params['max_leaf_nodes'])
            "n_estimators" : int(params["n_estimators"]),
            "max_depth" : int(params["max_depth"]),
            'max_features': int(params['max_features']),
            # "min_samples_split" : float(params["min_samples_split"]),
            # "min_samples_leaf" : int(params["min_samples_leaf"]),
        }
        # 选择模型
        model = RandomForestRegressor(**fit_params, random_state=42)
        model.fit(train_x, train_y)
        # 最小化测试集（- f1score）为目标
        # train_metric = model_metrics(model, train_x, train_y)
        test_metric = model_metrics(model, test_x, test_y)
        loss = - test_metric
        with open('./RFR-fitness.txt','a+') as fd:
            fd.write(str(loss) + '\n')
        return {"loss": loss, "status": STATUS_OK}

    # 参数空间
    # n_estimators = 80,
    # max_features=3
    #  ,max_depth=14
    #  ,min_samples_split=2
    #  ,min_samples_leaf=1
    space = {
        'n_estimators': hp.quniform('n_estimators', 1, 200, 1),
        'max_depth': hp.quniform('max_depth', 1, 50, 1),
        'max_features': hp.quniform('max_features', 1, 5, 1),
        # 'min_samples_split': hp.quniform('min_samples_split', 0.1, 1, 0.1),
        # 'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 30, 1)
    }
    # bayes优化搜索参数
    best_params = fmin(factory, space, algo=partial(anneal.suggest, ), max_evals=eval_iters, trials=Trials(),
                       return_argmin=True)
    # 参数转为整型
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["max_features"] = int(best_params["max_features"])
    # best_params["min_samples_split"] = float(best_params["min_samples_split"])
    # best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    return best_params


#  搜索最优参数
best_params = bayes_fmin(X_train, X_test, y_train, y_test, 500)
print(best_params)

rfr_reg = RandomForestRegressor(**best_params
                           , random_state=42)

rfr_reg.fit(X_train, y_train)
y_pred = rfr_reg.predict(X_test)

# 衡量线性回归的MSE 、 RMSE、 MAE、r2

mse_RFR = np.sum((np.array(y_test) - np.array(y_pred)) ** 2) / len(y_pred)
rmse_RFR = np.sqrt(mse_RFR)
mae_RFR = np.sum(np.absolute(np.array(y_test) - np.array(y_pred))) / len(y_test)
r_RFR = 1-mse_RFR/ np.var(y_test)#均方误差/方差


print("INFO: RMSE of RandomForestRegressor prediction is %.2f"%(rmse_RFR))
print("INFO: MAE of RandomForestRegressor prediction is %.2f"%(mae_RFR))
print("INFO: R2 of RandomForestRegressor prediction is %.2f"%(r_RFR))
# mse
# metrics.mean_squared_error(y_test, y_pred)
job_name = '{}'.format(datetime.datetime.now().strftime('%b%d-%H'))
joblib.dump(rfr_reg, './RandomForestRegressor-Bayesian-predict-%s.pkl'%(job_name))
