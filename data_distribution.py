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
skewness = {}
for col in train_df.columns:
    skewness[col] = train_df[col].skew()
    if train_df[col].skew()>1:
        train_df[col] = train_df[col].apply(np.sqrt)

skewness_1 = {}
for col in train_df.columns:
    skewness_1[col] = train_df[col].skew()
print(train_df.head())
print(train_df.info())
# univariate analysis
fig = plt.figure(figsize=(20, 12))

color=['grey','darkorange','darkviolet','turquoise','r','g','b', 'lightgreen', 'm', 'y',
'k','darkorange','lightgreen','plum', 'tan',
'khaki', 'pink', 'skyblue','lawngreen','salmon']
# 坐标系标签使用西文字体
ticklabels_style = {
    "fontname": "Times New Roman",
    "fontsize": 16,  # 小五号，9磅
}
i = 0
for column in train_df:
    sub = fig.add_subplot(2, 4, i + 1)
    sub.set_xlabel(column, fontsize=18, fontweight='bold')
    sub.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    train_df[column].plot(kind='hist', color=color[i], fontsize=18, grid=True)
    plt.grid(linestyle="--", alpha=0.8)
    if column=='XB':
        plt.xticks(**ticklabels_style)
        plt.xlabel('X$_{B}$')
    elif column=='X50':
        plt.xticks(**ticklabels_style)
        plt.xlabel('X$_{50}$')
    else:
        plt.xticks(**ticklabels_style)
    plt.yticks(**ticklabels_style)

    # ax.set_ylabel(fontsize=15)  # 设置y轴标签字体大小
    # ax.set_xlabel(fontsize=15)  # 设置x轴标签字体大小
    i = i + 1
plt.subplots_adjust(wspace=0.4, hspace =0.2) #调整子图间距
# plt.show()

plt.savefig(os.path.join(save_result, '特征分布情况0531.jpg'), dpi=1280, bbox_inches='tight', pad_inches=0.1)
# plt.show()
