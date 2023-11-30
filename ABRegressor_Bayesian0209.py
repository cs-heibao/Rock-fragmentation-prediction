import xgboost
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, Trials, STATUS_OK, Trials, anneal
from functools import partial
from hyperopt.fmin import fmin
from sklearn.metrics import f1_score,r2_score,mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# 特征数据归一化处理
from sklearn.preprocessing import StandardScaler

# 下面3行代码就是将数据集随即按照num_train分成训练集和测试集，数据量大，就分了两部分
train_df = pd.read_csv("train_0827.csv")
test_df = pd.read_csv("test1.csv")
train_df.drop('ID', axis=1, inplace=True)
test_df.drop('ID', axis=1, inplace=True)

X_train = train_df.drop(['X50'], axis=1)
y_train = train_df['X50']

X_test = test_df.drop(['X50'], axis=1)
Y_test = test_df['X50']



def model_metrics(model, x, y):
    """ 评估指标 """
    yhat = model.predict(x)
    # return r2_score(y, yhat)
    return np.sqrt(mean_squared_error(y, yhat))


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
            'learning_rate': float(params['learning_rate']),
            "n_estimators" : int(params["n_estimators"]),
            # "max_depth" : int(params["max_depth"]),
            # 'max_features': int(params['max_features']),
            # "min_samples_split" : int(params["min_samples_split"]),
            # "min_samples_leaf" : int(params["min_samples_leaf"]),
        }
        # 选择模型
        model = AdaBoostRegressor(**fit_params,random_state=42)
        model.fit(train_x, train_y)
        # 最小化测试集（- f1score）为目标
        test_metric = model_metrics(model, test_x, test_y)
        loss = test_metric
        with open('./ABR-Bayesian-fitness-RMSE.txt','a+') as fd:
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
        # 'max_depth': hp.quniform('max_depth', 1, 50, 1),
        'learning_rate': hp.quniform('learning_rate', 0.001, 1, 0.01),
        # 'max_features': hp.quniform('max_features', 1, 11, 1),
        # 'min_samples_split': hp.quniform('min_samples_split', 1, 30, 1),
        # 'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 30, 1)
    }
    # bayes优化搜索参数
    best_params = fmin(factory, space, algo=partial(anneal.suggest, ), max_evals=eval_iters, trials=Trials(),
                       return_argmin=True)
    # 参数转为整型
    best_params["n_estimators"] = int(best_params["n_estimators"])
    # best_params["max_depth"] = int(best_params["max_depth"])
    best_params["learning_rate"] = float(best_params["learning_rate"])
    # best_params["max_features"] = int(best_params["max_features"])
    # best_params["min_samples_split"] = int(best_params["min_samples_split"])
    # best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    return best_params


#  搜索最优参数
best_params = bayes_fmin(X_train, X_test, y_train, Y_test, 500)
# best_params["loss"] = 'quantile'
print(best_params)

abr_reg = AdaBoostRegressor(**best_params
                           ,random_state=42)

abr_reg.fit(X_train, y_train)
y_pred_train = abr_reg.predict(X_train)
y_pred = abr_reg.predict(X_test)

# training dataset
# 衡量线性回归的MSE 、 RMSE、 MAE、r2
rmse_ABR_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_ABR_train = mean_absolute_error(y_train, y_pred_train)
r_ABR_train = r2_score(y_train, y_pred_train)#均方误差/方差


# testing dataset
# 衡量线性回归的MSE 、 RMSE、 MAE、r2
rmse_ABR = np.sqrt(mean_squared_error(Y_test, y_pred))
mae_ABR = mean_absolute_error(Y_test, y_pred)
r_ABR = r2_score(Y_test, y_pred)#均方误差/方差
# abr_reg.score(X_test,y_test)
print("INFO: RMSE of AdaBoostRegressor prediction is %.2f"%(rmse_ABR))
print("INFO: MAE of AdaBoostRegressor prediction is %.2f"%(mae_ABR))
print("INFO: R2 of AdaBoostRegressor prediction is %.2f"%(r_ABR))
# mse
# metrics.mean_squared_error(y_test, y_pred)
job_name = '{}'.format(datetime.datetime.now().strftime('%b%d-%H'))
joblib.dump(abr_reg, './AdaBoostRegressor-Bayesian-predict-%s.pkl'%(job_name))
