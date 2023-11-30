import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
from matplotlib.pyplot import MultipleLocator

save_result='./Analysis_Result'
font_path = '/home/jie/Downloads/times.ttf'
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
train_df = pd.read_csv("train_0827.csv")
test_df = pd.read_csv("test1.csv")
train_df.drop('ID', axis=1, inplace=True)
test_df.drop('ID', axis=1, inplace=True)


# np.sqrt用于转换右偏特征- “慷慨”和“腐败感知”。结果，这两个特征变得更加正态分布。
for feature in [ 'S/B', 'T/B', 'Pf']:
       # fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
       # train_df[feature].hist(ax=ax_before)
       train_df[feature] = train_df[feature].apply(np.sqrt)
       # train_df[feature].hist(ax=ax_after)

X_train = train_df.drop(['X50'], axis=1)
y_train = train_df['X50']

X_test = test_df.drop(['X50'], axis=1)
y_test = test_df['X50']

data = pd.read_csv("2022-10-10.csv")

target = np.array(data['Target'])
GBoost_BOA_result = np.array([0.40566,0.64001,0.39765,0.29185,0.19117,0.35,0.19,0.23927,0.20344,0.21882,0.2])
R = dict()
features = X_test.columns
for para in features:
       up = 0
       down_1 = 0
       down_2 = 0
       mean = X_test[para].mean()
       mean_y = GBoost_BOA_result.mean()
       for i in range(X_test[para].size):
              up+= (X_test[para][i] - mean) *(GBoost_BOA_result[i]-mean_y)
              down_1+=(X_test[para][i] - mean)**2
              down_2 += (GBoost_BOA_result[i]-mean_y) ** 2
       R[para] = up/np.sqrt((down_1*down_2))
# 坐标系标签使用西文字体
ticklabels_style = {
    "fontname": "Times New Roman",
    "fontsize": 12,  # 小五号，9磅
}
plt.figure(figsize=(8, 5))
colors = ['r', 'g', 'c', 'm', 'b', 'y', '#d62728', '#7f7f7f', '#bcbd22', '#17becf']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f', '#bcbd22', '#e377c2']
x_width = range(0, len(R))
p1 = plt.barh(x_width, list(R.values()), 0.35, color=colors)
plt.bar_label(p1, label_type='edge', fmt='%.2f')
plt.yticks(range(0, len(R)), X_train.columns)
plt.xlabel('Sensitivity Relevancy Factor (SRF)', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.yticks(**ticklabels_style)
plt.xticks(**ticklabels_style)
plt.yticks([0, 1, 2, 3, 4, 5, 6], labels=['S/B', 'H/B', 'B/D', 'T/B', 'Pf', 'X$_{B}$', 'E'])
plt.grid(linestyle="--", alpha=0.3)
plt.savefig(os.path.join(save_result, '敏感性分析0531.jpg'), dpi=1280, bbox_inches='tight')

# plt.show()
print('ok')