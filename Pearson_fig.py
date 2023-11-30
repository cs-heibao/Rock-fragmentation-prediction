## for data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import os

font_path = '/home/jie/Downloads/times.ttf'
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
save_result = './Analysis_Result/'

train_df = pd.read_csv("train_0827.csv")
test_df = pd.read_csv("test1.csv")


train_df.drop('ID', axis=1, inplace=True)
test_df.drop('ID', axis=1, inplace=True)
print(train_df.head())
print(train_df.info())
ticklabels_style = {
    "fontname": "Times New Roman",
    "fontsize": 12,  # 小五号，9磅
}

corr_matrix = train_df.corr(method="pearson")
sns.set_theme(font='Times New Roman',font_scale=1.0)
# sns.set(font_scale=1, font=ticklabels_style)
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.yticks(**ticklabels_style)
plt.xticks(**ticklabels_style)
plt.title("Pearson Correlation", ticklabels_style,fontweight='bold')
plt.savefig(os.path.join(save_result, '皮尔逊相关系数0909.jpg'), dpi=1280, bbox_inches='tight', pad_inches=0.1)
plt.show()
