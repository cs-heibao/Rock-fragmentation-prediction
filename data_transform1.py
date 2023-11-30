## for data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from scipy.stats import norm
import scipy.stats as st
import os
def normfun(x, mu, sigma):
    pdf = np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf
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

kurtosis = {}
skewness = {}
for col in train_df.columns:
    kurtosis[col] = train_df[col].kurtosis()
    skewness[col] = train_df[col].skew()
# univariate analysis
fig = plt.figure(figsize=(20, 12))
# fig , ax = plt.subplots(figsize=(20,12))

color=['grey','darkorange','darkviolet','turquoise','r','g','b', 'lightgreen', 'm', 'y',
'k','darkorange','lightgreen','plum', 'tan',
'khaki', 'pink', 'skyblue','lawngreen','salmon']
# 坐标系标签使用西文字体
ticklabels_style = {
    "fontname": "Times New Roman",
    "fontsize": 16,  # 小五号，9磅
}
# np.sqrt用于转换右偏特征- “慷慨”和“腐败感知”。结果，这两个特征变得更加正态分布。
i = 0
for column in ['S/B', 'T/B', 'Pf']:
    sub = fig.add_subplot(2, 3, i + 1)
    sub.set_xlabel(column, fontsize=18, fontweight='bold')
    sub.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    sub.set_title('Before square root', fontsize=15, fontweight='bold')  # 设置图名字体大小
    # train_df[column].plot(kind='hist', color=color[i], fontsize=18, grid=True)
    sns.distplot(train_df[column], fit=norm, color=color[i],norm_hist=False)
    (mu, sigma) = norm.fit(train_df[column])
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], fontsize=15,
               loc='best')
    # plt.xlim([0.0, 1.0])
    x = np.arange(np.array(train_df[column]).min(), np.array(train_df[column]).max(), 0.01)
    mean = np.array(train_df[column]).mean()
    std = np.array(train_df[column]).std()
    y = normfun(x, mean, std)
    # y1 = normfun(x, mu1, sigma1)
    # plt.plot(x, y)
    # plt.plot(x, y1, 'r')
    plt.grid(linestyle="--", alpha=0.8)
    plt.yticks(**ticklabels_style)
    plt.xticks(**ticklabels_style)
    # ax.set_ylabel(fontsize=15)  # 设置y轴标签字体大小
    # ax.set_xlabel(fontsize=15)  # 设置x轴标签字体大小
    i = i + 1
    # plt.show()
    # print('ok')

i =3
ranges = [[1.0, 1.75], [0.5, 4.67], [0.22, 1.26]]
for column in [ 'S/B', 'T/B', 'Pf']:
    sub = fig.add_subplot(2, 3, i + 1)
    sub.set_xlabel(column, fontsize=18, fontweight='bold')
    sub.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    sub.set_title('After square root', fontsize=15, fontweight='bold')  # 设置图名字体大小
    train_df[column] = train_df[column].apply(np.log)
    # train_df[column].plot(kind='hist', color=color[i], fontsize=18, grid=True)
    sns.distplot(train_df[column], fit=norm, color=color[i],norm_hist=False)
    (mu, sigma) = norm.fit(train_df[column])
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],fontsize=15,
               loc='best')
    # x = np.arange(np.array(train_df[column]).min(), np.array(train_df[column]).max(), 0.01)
    # mean = np.array(train_df[column]).mean()
    # std = np.array(train_df[column]).std()
    # y = normfun(x, mean, std)
    # y1 = normfun(x, mu1, sigma1)
    # plt.plot(x, y)

    plt.grid(linestyle="--", alpha=0.8)
    # plt.xlim(ranges[i-3])
    plt.yticks(**ticklabels_style)
    plt.xticks(**ticklabels_style)
    # ax.set_ylabel(fontsize=15)  # 设置y轴标签字体大小
    # ax.set_xlabel(fontsize=15)  # 设置x轴标签字体大小

    i = i + 1

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.25, hspace =0.2) #调整子图间距
plt.show()
# print('ok')
kurtosis1 = {}
skewness1 = {}
for col in train_df.columns:
    kurtosis1[col] = train_df[col].kurtosis()
    skewness1[col] = train_df[col].skew()
plt.savefig(os.path.join(save_result, '特征分布转换后1117.jpg'), dpi=1280, bbox_inches='tight', pad_inches=0.1)
