# -- coding:utf-8 --
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin

# from utils import is_number

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl

warnings.filterwarnings('ignore')
#%matplotlib inline
sns.set_style("white")

plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False


######################
## CV functions
######################
## AUC and f1 score with CV

def StratifiedKFold_func_with_features_sel(x, y,Num_iter=100,score_type = 'rmse'):
    min_max_scaler = preprocessing.MinMaxScaler()
    #x_data = min_max_scaler.fit_transform(x)
    # Hierarchical K-fold cross-validation
    acc_v = []
    acc_t = []
    # Each time K-fold 100 times!
    for i in range(Num_iter):
        # Each fold is random random_state=i
        skf = KFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            x_tr_normalized = min_max_scaler.fit_transform(x_tr)
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            x_te_normalized = min_max_scaler.fit_transform(x_te)
            y_te = y[te_idx]
            # Define model hyperparameters
            model = xgb.XGBRegressor(max_depth=4,learning_rate=0.2,reg_alpha=1)
            #Model fitting
            model.fit(x_tr_normalized, y_tr)
            pred = model.predict(x_te_normalized)
            train_pred = model.predict(x_tr_normalized)
            #Call sklearn's roc_auc_score and f1_score to calculate related indicators
            ## Note that L here uses the predicted label value instead of the AUC for the
            # prediction probability. The reason is that this article focuses on the prediction
            # to distinguish between life and death. The use of the predicted label is equivalent
            # to the verification of the model's results when the threshold is determined to be 0.5.
            ## The AUC threshold segmentation point can be regarded as 1, 0.5, 0, respectively,
            # so that it can better reflect the difference in the distinguishing performance of
            # the features, and find the features that can contribute to the distinguishing degree.

            if score_type == 'rmse':
                acc_v.append(math.sqrt(np.mean(y_te-pred)**2))
                acc_t.append(math.sqrt(np.mean(y_tr-train_pred)**2))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))
    if x.shape[1]==2:
        drow_scatter_plot(x_tr[:,0],x_tr[:,1],y_tr,train_pred,'xgboost-train')
    # Returns the average
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]

def StratifiedKFold_func(x, y, Num_iter=1, model = xgb.XGBClassifier(max_depth=4,learning_rate=0.2,reg_alpha=1), score_type ='auc'):
    # The k-fold of the model outside the loop
    # Hierarchical K-fold cross-validation
    acc_v = []
    acc_t = []
    
    for i in range(Num_iter):
        skf = KFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]

            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)

            pred_Proba = model.predict_proba(x_te)[:,1]
            train_pred_Proba = model.predict_proba(x_tr)[:,1]

            if score_type == 'auc':
            	acc_v.append(roc_auc_score(y_te, pred_Proba))
            	acc_t.append(roc_auc_score(y_tr, train_pred_Proba))
            else:
            	acc_v.append(f1_score(y_te, pred))
            	acc_t.append(f1_score(y_tr, train_pred))
    # print(len(acc_t))

    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


def Compare_StratifiedKFold(x, y, Num_iter=100, model=xgb.XGBClassifier(max_depth=4, learning_rate=0.2, reg_alpha=1)):
    # The k-fold of the model outside the loop
    # Hierarchical K-fold cross-validation
    rmse_v = []
    rmse_t = []

    for i in range(Num_iter):
        skf = KFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x, y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]

            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)
            rmse_v.append(math.sqrt(np.mean(y_te - pred) ** 2))
            rmse_t.append(math.sqrt(np.mean(y_tr - train_pred) ** 2))


    # print(len(acc_t))

    return [np.mean(rmse_t), np.mean(rmse_v), np.std(rmse_t), np.std(rmse_v)]

################################
## Read data functions
###############################
def read_train_data(path_train):
    data_df = pd.read_excel(path_train, encoding='gbk', index_col=[0, 1])  # train_sample_375_v2 train_sample_351_v4
    data_df = data_df.groupby('PATIENT_ID').last()
    # data_df = data_df.iloc[:,1:]
    # data_df = data_df.set_index(['PATIENT_ID'])
    # data_df['年龄'] = data_df['年龄'].apply(lambda x: x.replace('岁', '') if is_number(x.replace('岁', '')) else np.nan).astype(float)
    # data_df['性别'] = data_df['性别'].map({'男': 1, '女': 2})
    # data_df['护理->出院方式'] = data_df['护理->出院方式'].map({'治愈': 0,'好转': 0, '死亡': 1})
    lable = data_df['出院方式'].values
    data_df = data_df.drop(['出院方式', '入院时间', '出院时间'], axis=1)
    data_df['Type2'] = lable
    data_df = data_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df = data_df.applymap(lambda x: x if is_number(x) else -1)
    # data_df = data_df.loc[:, data_df.isnull().mean() < 0.2]
    data_df = data_df.astype(float)

    return data_df

### is_number in the read data
def is_number(s):
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

### Data read and split

## calculate miss values by col
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

######################
## Plot functions
######################
def show_confusion_matrix(validations, predictions):
    LABELS = ['Survival','Death']
    matrix = metrics.confusion_matrix(validations, predictions)
    # plt.figure(dpi=400,figsize=(4.5, 3))
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc(labels, predict_prob,Moodel_name_i,fig,labels_name,k):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    #plt.figure()
    line_list = ['--','-']
    ax = fig.add_subplot(111)
    plt.title('ROC', fontsize=20)
    ax.plot(false_positive_rate, true_positive_rate,line_list[k%2],linewidth=1+(1-k/5),label=Moodel_name_i+'( AUC = %0.4f)'% roc_auc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.xlabel('FPR', fontsize=20)
    labels_name.append(Moodel_name_i+' AUC = %0.4f'% roc_auc)
    #plt.show()
    return labels_name


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def plot_decision_boundary(model, x_tr, y_tr):
    """画出决策边界和样本点

    :param model: 输入 XGBoost 模型
    :param x_tr: 训练集样本
    :param y_tr: 训练集标签
    :return: None
    """
    # x_ss = StandardScaler().fit_transform(x_tr)
    # x_2d = PCA(n_components=2).fit_transform(x_ss)

    coord1_min = x_tr[:, 0].min() - 1
    coord1_max = x_tr[:, 0].max() + 1
    coord2_min = x_tr[:, 1].min() - 1
    coord2_max = x_tr[:, 1].max() + 1

    coord1, coord2 = np.meshgrid(
        np.linspace(coord1_min, coord1_max, int((coord1_max - coord1_min) * 30)).reshape(-1, 1),
        np.linspace(coord2_min, coord2_max, int((coord2_max - coord2_min) * 30)).reshape(-1, 1),
    )
    coord = np.c_[coord1.ravel(), coord2.ravel()]

    category = model.predict(coord).reshape(coord1.shape)
    # prob = model.predict_proba(coord)[:, 1]
    # category = (prob > 0.99).astype(int).reshape(coord1.shape)

    dir_save = './decision_boundary'
    os.makedirs(dir_save, exist_ok=True)

    # Figure
    plt.close('all')
    plt.figure(figsize=(7, 7))
    custom_cmap = ListedColormap(['#EF9A9A', '#90CAF9'])
    plt.contourf(coord1, coord2, category, cmap=custom_cmap)
    plt.savefig(pjoin(dir_save, 'decision_boundary1.png'), bbox_inches='tight')
    plt.scatter(x_tr[y_tr == 0, 0], x_tr[y_tr == 0, 1], c='yellow', label='Survival', s=30, alpha=1, edgecolor='k')
    plt.scatter(x_tr[y_tr == 1, 0], x_tr[y_tr == 1, 1], c='palegreen', label='Death', s=30, alpha=1, edgecolor='k')
    plt.ylabel('Lymphocytes (%)')
    plt.xlabel('Lactate dehydrogenase')
    plt.legend()
    # plt.savefig(pjoin(dir_save, 'decision_boundary2.png'), dpi=500, bbox_inches='tight')
    plt.show()

def plot_3D_fig(X_data):
    cols = ['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']
    X_data = X_data.dropna(subset=cols,how='all')
    col = 'Type2'
    data_df_sel2_0 = X_data[X_data[col]==0]
    data_df_sel2_1 = X_data[X_data[col]==1]
    
    # fig = plt.figure(dpi=400,figsize=(10, 4))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    i= 2;j= 0;k= 1; # 120 201
    ax.scatter(data_df_sel2_0[cols[i]], data_df_sel2_0[cols[j]], data_df_sel2_0[cols[k]], c=data_df_sel2_0[col],cmap='Blues_r',label='Cured', linewidth=0.5)
    ax.scatter(data_df_sel2_1[cols[i]], data_df_sel2_1[cols[j]], data_df_sel2_1[cols[k]], c=data_df_sel2_1[col], cmap='gist_rainbow_r',label='Death',marker='x', linewidth=0.5)
 
    cols_en = ['Lactate dehydrogenase','Lymphocyte(%)','High-sensitivity C-reactive protein','Type of Survival(0) or Death(1)']
    ax.set_zlabel(cols_en[k])  # 坐标轴
    ax.set_ylabel(cols_en[j])
    ax.set_xlabel(cols_en[i])
    fig.legend(['Survival','Death'],loc='upper center')
    # plt.savefig('./picture_2class/3D_data_'+str(i)+str(j)+str(k)+'_v6.png')
    plt.show()

    # Set y values of data to lie between 0 and 1
    def normalize_data(dataset, data_min, data_max):
        data_std = (dataset - data_min) / (data_max - data_min)
        test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
        return test_scaled

    # Import and pre-process data for future applications
    def preprocess_data(train_dataframe):
        dataset = train_dataframe
        dataset = dataset.astype('float32')
        shape = dataset.shape[1]
        max_test = np.max(dataset[:, shape - 1])
        min_test = np.min(dataset[:, shape - 1])
        scale_factor = max_test - min_test
        max = np.empty(shape)
        min = np.empty(shape)
        # Create training dataset
        for i in range(0, shape):
            min[i] = np.amin(dataset[:, i], axis=0)
            max[i] = np.amax(dataset[:, i], axis=0)
            dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

        train_data = dataset[:, 0:shape - 1]
        train_labels = dataset[:, shape - 1]

        return train_data, train_labels, scale_factor


def data_read_and_split():
    train_dataframe = pd.read_excel("data/air_permeability.xlsx")
    train_dataframe = train_dataframe.drop("Sample Number",axis=1)
    # get the feature name
    cols = list(train_dataframe)
    print(cols)
    feature_name = cols[:-1]
    actual_value = cols[-1]
    x_data = train_dataframe[feature_name]
    y_data = train_dataframe[actual_value]
    print(x_data)
    return x_data,y_data,feature_name
#data_read_and_split()
def drow_scatter_plot(x,y,z_actual,z_predict,model_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z_actual,color='red',marker ='x',label='Actual')
    ax.scatter3D(x, y, z_predict, color='green',marker ='^', label='Predicted')
    #plt.title("simple 3D scatter plot")
    plt.legend(loc='upper right')
    ax.set_xlabel('Yarn Count')
    ax.set_ylabel('Stitch density')
    ax.set_zlabel('Air Permeability')
    plt.savefig(model_name)
    plt.show()

