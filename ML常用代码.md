# Def
- Feature 特征变量，也叫自变量，是样本可以观测到的特征，通常是模型的输入。
- Label 标签，也叫目标变量，需要预测的变量，通常是模型的标签或者输出。
- Train Data 训练数据，**有标签**的数据，**由举办方提供**。
- Train Set 训练集，从Train Data中分割得到，用于训练模型（常用于交叉验证）。
- Valid Set 验证集，**从Train Data中分割得到**，为了能找出效果最佳的模型，使用各个模型对验证集数据进行预测，并记录模型准确率。选出效果最佳的模型所对应的参数，即用来调整模型参数（常用于交叉验证）。
- Test Data 测试数据，通过训练集和验证集得出最优模型后，使用测试集进行模型预测。用来衡量该最优模型的性能和分类能力，**标签未知**，是比赛用来评估得分的数据，**由举办方提供**

# 常用数据科学方法
- 日常工作：Logistic 回归
- 国家安全领域：神经网络
- 简单的线性与非线性分类器
- 功能强大的集成方法：神经网络>SVM，多层感知机>带核函数的SVM

# 常用资源
- 学习资源: Stack Overflow Q&A, blog, personal project, Arxiv
- 开源数据: [Kaggle dataset aggregators](https://www.kaggle.com/datasets)

```Python
###################################库#############################################
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import pickle
import graphviz
from time import time
import random
import pydotplus 
import os
import re
import jieba
from numpy.random import randn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  #'all'|'last'|'last_expr'|'none'

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

input_path = '../input/'
submi_path = '../submision/'

###工具函数########################################
def age_group(age):  #年龄分组
    #用法：ori_data_28['age_group'] = ori_data_28.age.apply(age_group)
    if (age>=0) and (age<=10):
        return 0
    elif (age>10) and (age<=20):
        return 1
    elif (age>20) and (age<=30):
        return 2
    elif (age>30) and (age<=40):
        return 3
    elif (age>4) and (age<=50):
        return 4
    elif (age>50) and (age<=60):
        return 5
    elif (age>60) and (age<=70):
        return 6
    elif (age>70) and (age<=80):
        return 7
    elif (age>80) and (age<=90):
        return 8
    elif (age>90):
        return 9
#编码，one-hot，归一化
def encode_count(df, encoder_list):
    lbl = LabelEncoder()
    for i in range(0, len(encoder_list)):
        str_column_name = encoder_list[i]
        df[[str_column_name]] = lbl.fit_transform(df[[str_column_name]])
    return df
def encode_onehot(df, oneHot_list):
    for i in range(0, len(oneHot_list)):
        str_column_name = oneHot_list[i]
        feature_df = pd.get_dummies(df[str_column_name], prefix=str_column_name)
        df = pd.concat([df.drop([str_column_name], axis=1), feature_df], axis=1)
    return df
def normalize(df, normalize_list):
    scaler = StandardScaler()
    for i in range(0, len(normalize_list)):
        str_column_name = normalize_list[i]
        df[[str_column_name]] = scaler.fit_transform(df[[str_column_name]])
    return df
#制作统计特征
def feat_nunique(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].nunique().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_mode(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].mode().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_count(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].count().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_sum(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].sum().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_mean(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].mean().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_max(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].max().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
def feat_min(df_tar, df_fea, grou, stati, name, na=0):
    add_tem = df_fea.groupby(grou)[stati].min().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar
#特征选择
def Tree_feature_select(X_train, y_train, X_val, y_val, X_test, sel_num): 
     ##最简单最常用
    clf = lgb.LGBMClassifier(n_estimators=10000,
                             learning_rate=0.06,
                             max_depth=5,
                             num_leaves=30,
                             objective='binary',
                             subsample=0.9,
                             sub_feature=0.9,
                            )
    clf = clf.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
                  early_stopping_rounds=100, verbose = 500, 
                 )

    
    if type(X_train) is pd.core.frame.DataFrame:
        print('type(X_train) is DataFrame')
        feat_impo = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
        sel_list = [feat[0] for feat in feat_impo[0: sel_num]]
        X_train_sel = X_train[sel_list]
        X_val_sel = X_val[sel_list]
        X_test_sel = X_test[sel_list]
    elif (type(X_train) is np.ndarray) or (type(X_train) is csr_matrix):
        if type(X_train) is np.ndarray:
            print('type(X_train) is ndarray')
            feat_impo = sorted(zip(range(0, len(X_train[0])), clf.feature_importances_), key=lambda x: x[1], reverse=True)
            
        elif type(X_train) is csr_matrix:
            print('type(X_train) is csr_matrix')
            feat_impo = sorted(zip(range(0, X_train.get_shape()[1]), clf.feature_importances_), key=lambda x: x[1], reverse=True)

        sel_list = [feat[0] for feat in feat_impo[0: sel_num]]
        X_train_sel = X_train[:, sel_list]
        X_val_sel = X_val[:, sel_list]
        X_test_sel = X_test[:, sel_list]
    else:
        print('暂时不支持')
        return
        
    print("sel_num is:", sel_num)
    return feat_impo, X_train_sel, X_val_sel, X_test_sel
def RFECV_feature_sel(X_train, y_train, X_val, X_test):  #递归特征消除
    ##天池的大佬经常使用！！！
    ###clf可以随便换！！！
    clf = lgb.LGBMClassifier()
    selor = RFECV(clf, step=1, cv=3)
    selor = selor.fit(X_train, y_train)

    X_train_sel = selor.transform(X_train)
    X_val_sel = selor.transform(X_val)
    X_test_sel = selor.transform(X_test)
    
    return selor, X_train_sel, X_val_sel, X_test_sel
#降维
def PCA_decomposition(X_train, X_test, num):  
    ##注pca不支持稀疏输入
    pca = PCA(num)
    pca.fit(X_train)
    low_X_train = pca.transform(X_train) 
    low_X_test = pca.transform(X_test) 

    return pca, low_X_train, low_X_test
#cnn提取特征
def get_activation(nn_model, layer, X_origin):  
    get_activations = bk.function([nn_model.layers[0].input, bk.learning_phase()], [nn_model.layers[layer].output])
    activations=get_activations([X_origin, 0])
    return activations
#jieba词典调整
def __jieba_expend_dict():
    ###增加词典
    jieba.add_word('背骶尾部')
    ###调整词典
    jieba.suggest_freq('撞到', True)
    jieba.suggest_freq(('致', '右'), tune=False)
    pass
__jieba_expend_dict()

###其他工具函数
pickle.dump(ori_data_28, open(input_path + 'ori_data_28.pkl', 'wb'))
ori_data_28 = pickle.load(open(input_path + 'ori_data_28.pkl', 'rb'))
df_testInfor.to_csv(input_path + 'df_testInfor.csv', index=False)
df_basicInfor = pd.read_csv(input_path + 'df_basicInfor.csv')



###模型训练预测#################################
def clf_train(X_train, y_train, X_val, y_val):
    clf = lgb.LGBMClassifier(n_estimators=10000,
                                   learning_rate=0.06,
                                   max_depth=5,
                                   num_leaves=30,
                                   objective='binary',
                                   subsample=0.8,
                                   sub_feature=0.8,
#                                    class_weight='balanced',  #设置样本平衡；好像不要会更好
                                   )
    clf.fit(X_train, y_train, 
            eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
            early_stopping_rounds=100, verbose = 5000,
            )
    feat_impo = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    return clf, feat_impo 
def clf_predict(clf, X_val):
    y_pred = clf.predict_proba(X_val, num_iteration=clf.best_iteration)
    
    return y_pred
def clf_evaluate(y_test_class, y_pred_prob, Threshold=0.5):
    y_pred_prob = y_pred_prob[:, 1]
    y_pred_class = [1 if y>=Threshold else 0 for y in y_pred_prob]

    matrix = confusion_matrix(y_test_class, y_pred_class)
    print(matrix)
    print('f1 score：', f1_score(y_test_class, y_pred_class))  
    print('precision：', precision_score(y_test_class, y_pred_class))
    print('recall：', recall_score(y_test_class, y_pred_class))
    print('accuracy：', accuracy_score(y_test_class, y_pred_class))
    print('AUC：', roc_auc_score(y_test_class, y_pred_prob))



start_time = time()

clf, feat_impo = clf_train(X_train_, y_train_, X_val_, y_val_)
y_pred = clf_predict(clf, X_val_)
clf_evaluate(y_val_, y_pred)

print('训练预测的时间为:', int(time() - start_time))
 
 ```
