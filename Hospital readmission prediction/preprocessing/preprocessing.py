import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,  StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
h2o.init()

def imputation(config, df):
    imp_type = config['imp_type']
    if imp_type['mean']:
        df.fillna(df.mean(), inplace=True)
    elif imp_type['median']:
        df.fillna(df.median(), inplace=True)
    else:
        print("MICE Impute is under construction...")
    return df

def outlier_heal(config, data):
    outlier_heal = config[4]['outlier_heal']['eif']
    
    if outlier_heal:
        iforest = IsolationForest(n_estimators=200, max_samples='auto', 
                          contamination=0.03, max_features=.8, 
                          bootstrap=False, n_jobs=-1, random_state=42)
        feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'charttime', 
                                          'subject_id', 'hadm_id']]
        #print(data[feats].dtypes)
        
        data.fillna(data.mean(), inplace=True)
        pred= iforest.fit_predict(data[feats])
        data['scores']=iforest.decision_function(data[feats])
        data['original_paper_score']= [-1*s + 0.5 for s in data['scores']]
        data['anomaly_label']=pred

        #print(data[data.anomaly_label==-1].shape)
        data = data[data['original_paper_score']<=0.52]
        drps = ['scores', 'original_paper_score','anomaly_label']
        data.drop(drps, axis=1, inplace=True)
        return data
    else: 
        
        print('just clean data.. Normalization and advanced anomaly healing will presnt on next version ... change config setting ')


def outlier_heal_eif(config, data):
    outlier_heal = config[4]['outlier_heal']['eif']
    
    if outlier_heal:
        h2o_df = h2o.H2OFrame(data)

        # Set the predictors
        feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'charttime', 
                                                  'subject_id', 'hadm_id']]

        # Define an Extended Isolation forest model
        eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                                  ntrees = 200,
                                                  sample_size = 256,
                                                  extension_level = len(feats) - 1)

        # Train Extended Isolation Forest
        eif.train(x = feats,
                  training_frame = h2o_df)

        # Calculate score
        eif_result = eif.predict(h2o_df)

        # Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
        # or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
        anomaly_score = eif_result["anomaly_score"]
        eif_df = eif_result.as_data_frame()
        data = pd.concat([data, eif_df], axis=1)
        data = data[data.anomaly_score<.7]
        #print(data[data.anomaly_label==-1].shape)
        
        drps = ['anomaly_score','mean_length']
        data.drop(drps, axis=1, inplace=True)
        return data
    else: 
        
        print('just clean data.. Normalization and advanced anomaly healing will presnt on next version ... change config setting ')

     
    
    
        
def encode(config, frst_vst, categories):
    cat_encoding_le = config[3]['cat_encoding']['le']
    if cat_encoding_le:
        le = LabelEncoder()
        for c in categories:
            frst_vst[c] = le.fit_transform(frst_vst[c].astype(str))
        return frst_vst
    #elif cat_encoding== 'woe':
        ### WOE code
    

def missing_rem(df):
    missingTr = df.isnull().sum()/len(df)
    missingTr = missingTr[missingTr>=0]
    missingTr = missingTr.sort_values()
    missingTr = pd.DataFrame({'col':missingTr.index, '%null':missingTr.values})
    drps = missingTr[missingTr['%null']>.95].col.values
    df.drop(drps, axis=1, inplace=True)
    return df
    
def clean_temporals(visit):
    visit = visit[visit['admittime']< visit['dischtime']]
    return visit

def clean_events_items(df, event_feats):
    for c in event_feats:
        df = df[(df[c]<999999) |(df[c].isnull())]
        #print(df.describe())
        return df

def prep_vsts_dts(df):
    
    
    df.admittime = pd.to_datetime(df.admittime)
    df.dischtime = pd.to_datetime(df.dischtime)
    df.dod = pd.to_datetime(df.dod)
    df.deathtime = pd.to_datetime(df.deathtime)
    return df

def extract_temporal_feats(frst_vst):
    frst_vst['admittime_day']=frst_vst.admittime.dt.day
    frst_vst['admittime_hour']=frst_vst.admittime.dt.hour
    frst_vst['dischtime_day']=frst_vst.admittime.dt.day
    frst_vst['dischtime_hour']=frst_vst.admittime.dt.hour
    
    frst_vst['visitlos'] =   (frst_vst['dischtime'] - frst_vst['admittime']) / np.timedelta64(1, 'D')
    frst_vst['visit_los_hrs'] = frst_vst.visitlos.astype('float32') * 24
    #frst_stay['los_hrs'] = frst_stay.los.astype('float32') * 24
    #frst_stay['lostostay']= frst_stay['lostostay'].dt.total_seconds() / 3600
    return frst_vst


    