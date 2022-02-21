import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import random 
import gc
from sklearn.preprocessing import LabelEncoder,  StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics  import * 
import tensorflow as tf
tf.keras.utils.set_random_seed(
    2022
)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from pathlib import Path
import shap

sys.path.append(os.path.abspath("../generate_dataset"))


def fallback_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except:
        return 0.5


def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

    

def build_models(config, data , model= 'xgb', tune= False, cv = 5, target_='30', test_size= .2):
    train, test = train_test_split(data, test_size=.2, random_state=42)
    print('train and test shapes:' , train.shape, test.shape)

    target_ntw = train['READMISSION-NTW']
    target_test_ntw = test['READMISSION-NTW']

    target7 = train['READMISSION-7']
    target_test7 = test['READMISSION-7']

    target30 = train['READMISSION-30']
    target_test30 = test['READMISSION-30']
    
    if target_ == '30':
        target = target30
        target_test = target_test30
    
    elif  target_ == '7':
        target = target7
        target_test = target_test7
        
    else:
        target = target_ntw
        target_test = target_test_ntw
          
    
    feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'charttime', 
                                          'subject_id', 'hadm_id']]
               
        
    if model=='XGB':
            
        xgb_params = {
            'objective':'binary:logistic', 
            'max_depth': 6, 
            'learning_rate': 0.01, 
            'booster':'gbtree', 
            'eval_metric': 'auc', 
            'max_leaves': 20, 
            'colsample_bytree': 0.5, 
            'subsample':0.8, 
            'lambda': 4, 
             'alpha':2, 
              }
        if config[9]['GPU']:
            print('GPU activated on config, run xgb on GPU...')
        
            xgb_params['gpu_id'] = 0
            xgb_params['tree_method'] = 'gpu_hist'
        
        if target_ == '7':
            xgb_params[ 'scale_pos_weight'] = 9
        elif target_ == '30':
            xgb_params[ 'scale_pos_weight'] = 7
        else:
            xgb_params[ 'scale_pos_weight'] = 3

        #start = timeit.default_timer()

        xgb_scores = []
        oof_xgb = np.zeros(len(train))
        pred_xgb = np.zeros(len(test))
        importances = pd.DataFrame()
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=4242)

        for fold_, (train_ind, val_ind) in enumerate(folds.split(train[feats], target)):
            print('fold : ', fold_)
            trn_data = xgb.DMatrix(data=train[feats].iloc[train_ind], label=target.iloc[train_ind])
            val_data = xgb.DMatrix(data= train[feats].iloc[val_ind], label=target.iloc[val_ind])
            xgb_model = xgb.train(xgb_params, trn_data, num_boost_round=5000, evals=[(trn_data, 'train'), (val_data, 'test')], verbose_eval=100, early_stopping_rounds=100)
            oof_xgb[val_ind] = xgb_model.predict(xgb.DMatrix(train[feats].iloc[val_ind]),  ntree_limit= xgb_model.best_ntree_limit)
            print(roc_auc_score(target.iloc[val_ind], oof_xgb[val_ind]))
            xgb_scores.append(roc_auc_score(target.iloc[val_ind], oof_xgb[val_ind]))
            importance_score = xgb_model.get_score(importance_type='gain')
            importance_frame = pd.DataFrame({'Importance': list(importance_score.values()), 'Feature': list(importance_score.keys())})
            importance_frame['fold'] = fold_ +1
            importances = pd.concat([importances, importance_frame], axis=0, sort=False)
            pred_xgb += xgb_model.predict(xgb.DMatrix(test[feats]), ntree_limit= xgb_model.best_ntree_limit)/folds.n_splits

        print('xgb model auc:', np.mean(xgb_scores))
        score = np.mean(xgb_scores)
        score_test =  roc_auc_score(target_test, pred_xgb) 
        pred_rd = np.where(pred_xgb >= 0.5, 1, 0)
        f1 = f1_score(target_test, pred_rd)
        #stop = timeit.default_timer()
        #print('Time: ', stop - start) 
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(train[feats])

        shap.summary_plot(shap_values, train[feats]) 
        return score, score_test, f1
        
    elif model=='Logisticregression':
        train, test = train_test_split(data, test_size=.2, random_state=42)
        print('train and test shapes:' , train.shape, test.shape)

        target_ntw = train['READMISSION-NTW']
        target_test_ntw = test['READMISSION-NTW']

        target7 = train['READMISSION-7']
        target_test7 = test['READMISSION-7']

        target30 = train['READMISSION-30']
        target_test30 = test['READMISSION-30']

        if target_ == '30':
            target = target30
            target_test = target_test30

        elif  target_ == '7':
            target = target7
            target_test = target_test7

        else:
            target = target_ntw
            target_test = target_test_ntw
            
        feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'charttime', 
                                          'subject_id', 'hadm_id']]
        train.fillna(train[feats].mean(), inplace=True)
        test.fillna(test[feats].mean(), inplace=True)
        ss= RobustScaler()
        train = pd.DataFrame(ss.fit_transform(train[feats]), columns=feats)
        test = pd.DataFrame(ss.fit_transform(test[feats]), columns=feats)
        #train =train[feats].copy()
        #test = test[feats].copy()
        scores = []
        oof = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        y_le = target.values
        n_splits=cv
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_, (train_ind, val_ind) in enumerate(folds.split(train, y_le)):
            print('fold:', fold_)
            X_tr, X_test = train[feats].iloc[train_ind], train[feats].iloc[val_ind]
            y_tr, y_test = y_le[train_ind], y_le[val_ind]
            clf = LogisticRegression(C=100, max_iter=400,solver='liblinear', class_weight= 'balanced', random_state=2020)
            clf.fit(X_tr, y_tr)
            oof[val_ind]= clf.predict_proba(X_test)[:, 1]
            y = clf.predict_proba(X_tr)[:,1] 
            print('train:',roc_auc_score(y_tr, y),'val :' , roc_auc_score(y_test, (oof[val_ind])))
            print(20 * '-')
            scores.append(roc_auc_score(y_test, oof[val_ind]))
            test_preds += clf.predict_proba(test[feats])[:,1]/n_splits

        print('log reg  roc_auc=  ', np.mean(scores))
        score = np.mean(scores)
        score_test =  roc_auc_score(target_test, test_preds) 
        pred_rd = np.where(test_preds >= 0.5, 1, 0)
        f1 = f1_score(target_test, pred_rd)
        return score, score_test, f1
    
        
    elif model=='MLP':
        train, test = train_test_split(data, test_size=.2, random_state=42)
        print('train and test shapes:' , train.shape, test.shape)

        target_ntw = train['READMISSION-NTW']
        target_test_ntw = test['READMISSION-NTW']

        target7 = train['READMISSION-7']
        target_test7 = test['READMISSION-7']

        target30 = train['READMISSION-30']
        target_test30 = test['READMISSION-30']

        if target_ == '30':
            target = target30
            target_test = target_test30

        elif  target_ == '7':
            target = target7
            target_test = target_test7

        else:
            target = target_ntw
            target_test = target_test_ntw
        feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'READMISSION-90', 'charttime', 
                                          'subject_id', 'hadm_id', 'stay_id']]
        train.fillna(train[feats].mean(), inplace=True)
        test.fillna(test[feats].mean(), inplace=True)
        ss= StandardScaler()
        train = pd.DataFrame(ss.fit_transform(train[feats]), columns=feats)
        test = pd.DataFrame(ss.fit_transform(test[feats]), columns=feats)
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=142)
        auc_score = []
        oof_nn= np.zeros((len(train)))
        test_preds = np.zeros((len(test)))
        for fold_, (train_ind, test_ind) in enumerate(folds.split(train, target)):
        # create model
            print('fold :', fold_)
            trn_x, trn_y = train.iloc[train_ind], target.iloc[train_ind]
            val_x, val_y = train.iloc[test_ind], target.iloc[test_ind]
            model = Sequential()
            model.add(Dense(128, input_shape=(train.shape[1],),   activation= 'relu' ))
            model.add(Dropout(0.2))
            model.add(Dense(64,  activation= 'relu' ))
            model.add(Dropout(0.2))
            model.add(Dense(1,  activation= 'sigmoid' ))
            
            # Compile model
            model.compile(loss= 'binary_crossentropy' , optimizer= 'Adam' , metrics=[auc])
            # Fit the model
            cp = callbacks.ModelCheckpoint(filepath="cp.hdf5", monitor="val_auc",  verbose=0,
                save_best_only=False, save_weights_only=False,  mode="auto")
            es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,
                                         verbose=1, mode='max', baseline=None, restore_best_weights=True)
            rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                              patience=3, min_lr=1e-6, mode='max', verbose=1)
            if target_ == '7':
                wt = 15
            elif target_ == '30':
                wt = 7
            else:
                wt = 2
            #weight_for_0 = (1 / neg) * (total / 2.0)
            #weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0:1 , 1:wt}
            history = model.fit(trn_x, trn_y, validation_data=(val_x, val_y),callbacks= [cp, es, rlr], class_weight=class_weight, epochs=100, batch_size=64, verbose=1)
            val_preds = model.predict(val_x)
            print("AUC = {}".format(metrics.roc_auc_score(val_y,val_preds)))
            auc_score.append(metrics.roc_auc_score(val_y,val_preds))
            oof_nn[test_ind] =  val_preds.ravel()
            test_fold_preds = model.predict(test)
            test_preds += test_fold_preds.ravel()
            
        K.clear_session()
        gc.collect()
        print('AUC mean: ', np.mean(auc_score))
        score = np.mean(auc_score)
        test_preds/=cv
        score_test =  roc_auc_score(target_test, test_preds) 
        pred_rd = np.where(test_preds >= 0.5, 1, 0)
        f1 = f1_score(target_test, pred_rd)
        return score, score_test, f1


