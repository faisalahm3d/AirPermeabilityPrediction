# -- coding:utf-8 --
import pandas as pd
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from utils_features_selection import *
from mpl_toolkits import mplot3d

from xgboost import plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import svm
from sklearn import tree
import math
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

## features selection part
def features_selection():
    ## Read 100 data
    X_data_all_features,Y_data,x_col = data_read_and_split()

    # Construct a dataframe to store the importance information of features
    import_feature = pd.DataFrame()
    import_feature['col'] = x_col
    import_feature['xgb'] = 0
    # Repeat the test 100 times
    for i in range(100): # 50,150
        #Each experiment randomly divides 100 data into 0.7 training set and 0.3 test set, pay attention to random random_state=i
        ## The reason for this method is that because of the small sample size available, in order to generate different
        # training sample sets, the ranking of the importance of the features is more stable, so I chose such a method.
        ## Different samples are generated by different random seeds each time, so that the influence of the
        # abnormality of a small number of samples on the importance of features can be suppressed to a certain extent.
        x_train, x_test, y_train, y_test = train_test_split(X_data_all_features, Y_data, test_size=0.3, random_state=i)
        #Define model hyperparameters
        model = xgb.XGBRegressor(
                max_depth=4,
                learning_rate=0.2,
                reg_lambda=1,
                n_estimators=150,
                subsample = 0.9,
                colsample_bytree = 0.9)
        #
        model.fit(x_train, y_train)
        print("Accuracy: %.2f" %model.score(x_test,y_test))
        print("RMSE: %.2f"% math.sqrt(np.mean((model.predict(x_test) - y_test) ** 2)))
        #Cumulative feature importance
        import_feature['xgb'] = import_feature['xgb']+model.feature_importances_/100
    # Sort by feature importance in descending order
    import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
    print('Top 3 features:')
    print(import_feature.head())
    # Sort feature importances from GBC model trained earlier
    # Location information according to feature importance
    indices = np.argsort(import_feature['xgb'].values)[::-1]
    #Get the top 3 important feature locations
    Num_f = 5
    indices = indices[:Num_f]
    
    # Visualise these with a barplot
    # plt.subplots(dpi=400,figsize=(12, 10))
    plt.subplots(figsize=(6, 5))
    # g = sns.barplot(y=list(name_dict.values())[:Num_f], x = import_feature.iloc[:Num_f]['xgb'].values[indices], orient='h') #import_feature.iloc[:Num_f]['col'].values[indices]
    g = sns.barplot(y=import_feature.iloc[:Num_f]['col'].values[indices], x = import_feature.iloc[:Num_f]['xgb'].values[indices], orient='h') #import_feature.iloc[:Num_f]['col'].values[indices]
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=10)
    sns.despine() 
    # plt.savefig('feature_importances_v3.png')
    plt.show()
    # g.set_title("The mean feature importance of XGB models");
    # Get the importance value of the top 10 important features
    import_feature_cols= import_feature['col'].values

    # Draw feature pyramid
    num_i = 1
    val_score_old = 10000
    val_score_new = 10000
    while val_score_new <= val_score_old:
        val_score_old = val_score_new
        # Special order in order of importance
        x_col = import_feature_cols[:num_i]
        print(x_col)
        X_data = X_data_all_features[x_col]#.values
        ## Cross-validation
        print('5-Fold CV:')
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
        print("Train RMSE-score is %.4f ; Validation RMSE-score is %.4f" % (acc_train,acc_val))
        print("Train RMSE-score-std is %.4f ; Validation RMSE-score-std is %.4f" % (acc_train_std,acc_val_std))
        val_score_new = acc_val
        num_i += 1
        
    print('Selected features:',x_col[:-1])
    
    return list(x_col[:-1])


def Compare_with_other_method(sub_cols=['Yarn Count (tex)','Stitch density (loops/cm2)']):
    ## Read 351 data set (remove all samples with empty sub_cols from 375)
    X_data_all_features, Y_data, x_col = data_read_and_split()
    x_np = X_data_all_features[sub_cols]

    #Figure 4 for si illustrates the problem. If it is 50% off, SI cannot be drawn. Figure 4
    X_train, X_val, y_train, y_val = train_test_split(x_np, Y_data, test_size=0.3, random_state=6)
    Num_iter = 10
    #fig = plt.figure(figsize=(16, 8))
    labels_names = []

    '''
    #Define the comparison method under full features
    xgb_n_clf = xgb.XGBClassifier(
        max_depth=4
        ,learning_rate=0.2
        ,reg_lambda=1
        ,n_estimators=150
        ,subsample = 0.9
        ,colsample_bytree = 0.9
        ,random_state=0)
    tree_clf = tree.DecisionTreeClassifier(random_state=0,max_depth=4) #random_state=0,之前没加
    RF_clf1 = RandomForestClassifier(random_state=0,n_estimators=150,max_depth=4,)
    LR_clf = linear_model.LogisticRegression(random_state=0,C=1,solver='lbfgs')
    LR_reg_clf = linear_model.LogisticRegression(random_state=0,C=0.1, solver='lbfgs')
    
    

    
    
    i = 0
    
    Moodel_name = ['Multi-tree XGBoost with all features',
                   'Decision tree with all features',
                   'Random Forest with all features',
                   'Logistic regression with all features with regularization parameter = 1 (by default)',
                   'Logistic regression with all features with regularization parameter = 10',]
    for model in [xgb_n_clf,tree_clf,RF_clf1,LR_clf,LR_reg_clf]:
        print('Model:'+Moodel_name[i])
        #K-fold with f1 evaluation
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np.values, y_np.values,Num_iter,model, score_type ='f1')
        print('F1-score of Train:%.6f with std:%.4f \nF1-score of Validation:%.4f with std:%.6f '%(acc_train,acc_train_std,acc_val,acc_val_std))
        # K discount based on auc evaluation method
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np.values, y_np.values,Num_iter,model, score_type ='auc')
        print('AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f '%(acc_train,acc_train_std,acc_val,acc_val_std))

        #To draw si picture 4
        model.fit(X_train,y_train)
        pred_train_probe = model.predict_proba(X_train)[:,1]
        pred_val_probe = model.predict_proba(X_val)[:,1]
        #plot_roc(y_val, pred_val_probe,Moodel_name[i],fig,labels_names,i) # 为了画si图4中的test
        #plot_roc(y_train, pred_train_probe,Moodel_name[i],fig,labels_names,i) # 为了画si图4 train
        print('AUC socre:',roc_auc_score(y_val, pred_val_probe))
        
        i = i+1
    '''
    ## Comparison of three-character single-tree models
    #x_np_sel = x_np[sub_cols] #选择三特征
    ## The data set is divided for a single training of a single tree and an AUC graph is generated.
    # The division method is the same as before.
   # X_train, X_val, y_train, y_val = train_test_split(x_np_sel, y_np, test_size=0.3, random_state=6)

    #For comparison of three-feature models
    xgb_clf = xgb.XGBRegressor(max_depth=4,
                learning_rate=0.2,
                reg_lambda=1,
                n_estimators=150,
                subsample = 0.9,
                colsample_bytree = 0.9)
    tree_clf = tree.DecisionTreeRegressor(random_state=0,max_depth=3)
    RF_clf2 = RandomForestRegressor(random_state=0,n_estimators=1,max_depth=3,)
    #added by me
    LR_clf = linear_model.LinearRegression()
    LR_reg_clf = linear_model.LinearRegression()
    svm_clf = svm.SVR()
    svm_clf2 = svm.NuSVR()
    nn_clf = MLPRegressor(hidden_layer_sizes=(5,2),alpha=1e-5)

    i = 0
    Moodel_name = ['XGBoost',
                   'DT',
                   'RF',
                   'LR with RP = 1',
                   'LR with RP=10',
                   'MLP',
                   'SVM',
                   'SVM2']
    for model in [xgb_clf,tree_clf,RF_clf2,LR_clf,LR_reg_clf,nn_clf,svm_clf,svm_clf2]:
        print('Model '+Moodel_name[i])
        # f1 results
        # acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np_sel.values, y_np.values,Num_iter,model, score_type ='f1')
        # print('F1-score of Train:%.6f with std:%.4f \nF1-score of Validation:%.4f with std:%.6f '%(acc_train,acc_train_std,acc_val,acc_val_std))
        # auc results
        #acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(x_np_sel.values, y_np.values,Num_iter,model, score_type ='auc')
        #print('AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f '%(acc_train,acc_train_std,acc_val,acc_val_std))

        model.fit(X_train,y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        print("RMSE : ", math.sqrt(np.mean(pred_val-y_val)**2))
        print("MAE :", mean_absolute_error(y_val,pred_val))
        print("MSE ",mean_squared_error(y_val,pred_val))
        pred_score = model.score(X_val,y_val)
        print(" R = ", pred_score)

        #plot 3dimensional scatter
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(X_val.values[:,0],X_val.values[:,1], y_val, c=y_val, cmap='Greens')
        ax.scatter3D(X_val.values[:,0],X_val.values[:,1], pred_val, c=pred_val, cmap='Blues')
        plt.savefig(Moodel_name[i])
        plt.show()
        #plot_roc(y_val, pred_val_probe,Moodel_name[i],fig,labels_names,i)
        # plot_roc(y_train, pred_train_probe,Moodel_name[i-5],fig,labels_names,i)
        #print('AUC socre:',roc_auc_score(y_val, pred_val))
        i = i+1

#if __name__ == '__main__':
    
    ## 特征筛选
selected_cols = features_selection()
#single_tree()
    ## Compare Method
print('Compare with other methods')
Compare_with_other_method()