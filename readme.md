# **Introducing MiPy: a Benchmarking Framework for Machine Learning on MIMIC-IV DataBase**

The framwork builds a set of data processing and modelling tools on the MIMIC-IV database in Python. The MIMIC database is hosted at:
https://physionet.org/content/mimiciv/0.4/

We are starting the framework with Hospital and ICU Readmission Prediction and will add Mortality Predcition and other tasks in the future.

## **Quick Start**

1.	Access the data at https://physionet.org/content/mimiciv/0.4/

2.	Copy the required files to a folder called “data”. If the folder does not exists, create one in the root directory of the project.

Hospital readmission required files: 
  - admissions.csv.gz
  - d_labitems.csv.gz
  - labevents.csv.gz
  - patients.csv.gz

ICU readmission required files:
  - admissions.csv.gz
  - chartevents.csv.gz
  - d_items.csv.gz
  - d_labitems.csv.gz
  - icustays.csv.gz
  - labevents.csv.gz
  - patients.csv.gz

3.	Run the “reduce_mem.py” script from “preprocessing” to reduce the size of the “chartevents” and “labevents” files

4.	Create the baseline cohort by running the “main.py” file in “generate_dataset”. This will create a new folder (“erStays”). This step is generic to both readmission tasks. There is also a Python Notebook version of the main.py

5.	Edit the “configge.ipynb” file in the “resources” folder to match your requirements. A description of the parameters is provided below. Then run this file to generate the required "config.yml"

6.	Run “main.py” again for the full run of the system. 

## **Full guide: Hospital Readmission Task:**
     
The file configgen.ipython generates config.yml which is used for setting task parameters. Initiate your parameters as your use case definition. 

- Defult config:
     - data_dir: ../data
    `- subtask:
        atAdmit: 0
        atDischarge: 1
        isEarly: 0
        
    - wt_forrolling:
        dischwt_for_rolling:
          24: 1
          48: 0
        erwt_for_rolling:
          2: 0
          6: 0
          12: 1
    - cat_encoding:
        le: 1
        oh: 0
        woe: 0
    - outlier_heal:
        Advanced: 0
        MAD: 0
        eif: 1
    - imp_type:
        mean: 1
        median: 0
        mice: 0
    - model:
        Logisticregression: 1
        MLP: 0
        XGB: 1
    - target:
        '30': 1
        '7': 1
        ntw: 0
    - measure:
        accuracy: 0
        f1_score: 1
        recall: 0
    - GPU: 1
    - CV:
        5: 1
        10: 0`

  
  - During atAdmit prediction just static_feats will be used and labevents_feats can be extracted when runing config with atDischarge and isEarly readmission prediction
  
  - dischwt_for_rolling(discharge) and erwt_for_rolling(early) determines the time window for rolling mean of labevents features (e.g 12 for early and 24 for atDischarge)
  
  - For this release we just implemented le(label encoder) for categorical encoding
  
  - Mice imputation will be implemented during next release
  
  - By default roc_auc  and  f1_score will be generated on results 
  
  - Multi-selection parameters:
    subtask, model, target, measure
   
  - Uni-selection parameters: 
    wt_forrolling(window time for rolling), cat_encoding, outlier_heal, imp_type(imputation type)
  
  
  - Some parameters will be presented during next versions so don't use following parameters in config setting:
    'cat_encoding':  'oh' and  'woe'
    'imp_type': median' and  'mice'
    'dischwt_for_rolling':  48 hours 
    'erwt_for_rolling': 2 and  6 hours
    'outlier_heal': 'Advanced' and 'MAD'


Note: 

Label extraction will be performed at the beginning of the running program, building '../erVisits' folder by breaking data by subjects.
After running the first time there is no need to do it twice as the labeled data folder exists for next running and will not run twice unless you remove ../generate_dataset/erVisits folder.
 
Folder Structure: 
```
 sudo apt-get install tree
 
 tree /path/to/root
 ```



- Requirements: 

    - numpy==1.19.5, 
    - pandas==1.2.4,
    - scipy==1.6.2, 
    - scikit-learn==0.24.2,
    - matplotlib==3.3.4, 
    - seaborn==0.11.1,  
    - imbalanced-learn==0.8.0, 
    - tensorboard==2.7.0, 'tensorflow-gpu==2.5.0',  'tensorflow==2.7.0',  
    - tqdm==4.59.0, 
    - xgboost==1.3.3,
    - shap==0.39.0, 
    - keras==2.7.0, 
    - pyyaml== 6.0 (pip install pyyaml)
    - h2o-3.36.0.3


 - Some important features of next releases:
   - Advanced resampling methods for handling imbalanced data
   - Advanced outlier healing methods
   - Advanced feature selection methods for algorithms
   - Tuning module for algorithms
   - Separated interpratation module

### **Steps to Run:**

**Step 0.** The app will run in two phases 
1. Label Extraction 
2. Preprocessing and Model building and training. 
   
So run main.py to build '../erVisits' folder by breaking stays by subjects and extracting labels.
After extracting labels and building labeled data run configgen.ipynb and and set parameters and run main.py again to continue based on config.yml.

For running the program find the main python file:  ../generate_dataset/main.py and run:
```
python generate_datasets/main.py   
```
Data generation and training will be notified through running based on config setting.



**Step 1.** Start with following config and run algorithms on "atAdmit" subtask, in this case only static and demographic features will be used :
```
dict_file = dict_file = [{'data_dir': '../data'},
            {'subtask':{'isEarly' : 0, 'atDischarge':0, 'atAdmit':1}},
            {'wt_forrolling':{'erwt_for_rolling':{12:1, 6:0, 2:0}, 'dischwt_for_rolling': {24:1, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'eif':1, 'MAD':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```

**Step 2.** Run step 1 and add labevents features (takes more run time) by selecting: ``` 'isEarly' : 1 ``` or  ``` 'atDischarge':1 ```

   (During above steps if GPU is not avaialable,  set GPU:0 )


Note for running algorithms: 

    - XBG handles missing data automatically and is less sensitive to outliers 
    - Imputing scaling before running linear algorithms we used is mandatory
    - In this version we used statis parameters for all input data, just tune target weights to handle the imbalanced data. To get better results tuning module is recommended
    
    
Note for outlier healing:

    We presented Iso_Forest.ipynb to show how we performed outlier healing with isolation forest.
    The results show imputing nan values is a significant factor and mean impute will increase the outlier scores. 
    
    - By default outlier_healing function uses "Extended Isolation Forest" . 


## **ICU Readmission Task**
- MIMIC-IV data: 

   - data_dir: ../data (put MIMIC-IV data in this folder)
   - patients.csv.gz
   - admissions.csv.gz
   - icustays.csv.gz
   - chart_events_df.pkl
   - d_items.csv.gz
   - lab_events_df.pkl
   - d_labitems.csv.gz

   Note: For chartevents and labevents data used reduced version of the data in .pkl format(chart_events_df.pkl, lab_events_df.pkl)
   for this run:
   ```
   chartevents = pd.read_csv(chartevents.csv.gz)
   chart_events_df = reduce_mem_usage(chartevents)
   chart_events_df.to_pickle('chart_events_df.pkl') [about 15GB]
   
   labevents = pd.read_csv(labevents.csv.gz)
   lab_events_df = reduce_mem_usage(chartevents)
   lab_events_df.to_pickle('lab_events_df.pkl') [about 9GB]
   ``` 
    and put chart_events_df.pkl and lab_events_df.pkl on ../data folder. reduce_memory_usage function is avalable in following path:
    ```
    '/root/generate_dataset/utils.py'
    ```
    
    If memory usage is not imortant use original .csv.gz from MIMIC-IV and change following path :
    ```
    path_chevents = os.path.join(data_dir, 'chart_events_df.pkl')
    path_labevents = os.path.join(data_dir, 'lab_events_df.pkl')
    ```
    in functions: roll_chart_items, roll_lab_items form:    
    /root/generate_dataset/utils.py  to .csv.gz
    
    
    
    Defult config:
    - subtask:
        atAdmit: 1
        atDischarge: 0
        isEarly: 0
    - feat_set: (feature sets)
        chart_events_feats: 1
        labevents_feats: 1
        static_feats: 1
    - wt_forrolling:
        dischwt_for_rolling: 24 (at discharge time windows( for rolling,last dischwt_for_rolling hours )
        erwt_for_rolling: 12 (early prediction  time windows for rolling, first erwt_for_rolling hours )
    - cat_encoding: (categorical encoding method)
        le: 1
        oh: 0
        woe: 0
    - outlier_heal: (outlier healing method)
        Advanced: 0
        byAnomalyScore: 1
        normalization: 0
    - imp_type: (imputation type)
        mean: 1
        median: 0
        mice: 0
    - model:
        Logisticregression: 1
        MLP: 1
        XGB: 1
    - target:
        '30': 1
        '7': 1
        ntw: 1
    - measure:
        accuracy: 0
        f1_score: 1
        recall: 0
    
  
  - atAdmit just static_feats will be used 
  
  - chart_events_feats and labevents_feats can be extracted when runing config with atDischarge and isEarly 
  
  - dischwt_for_rolling(discharge) and erwt_for_rolling(early) determines the time window for rolling mean of chartevents and labevents features 
  
  - For this release we just implemented le(label encoder) for categorical encoding
  
  - by default roc_auc  and  f1_score will be generated on results 
  
  - Multi-selection parameters:
    subtask, feat_set, model, target, measure
   
  - Uni-selection parameters: 
    wt_forrolling(window time for rolling), cat_encoding, outlier_heal, imp_type(imputation type)
  
  
  - Some parameters will be presented during next versions so don't use following parameters in config setting:
    'cat_encoding':  'oh' and  'woe'
    'imp_type': median' and  'mice'
    'dischwt_for_rolling':  48 hours 
    'erwt_for_rolling': 2 and  6 hours
    'outlier_heal': 'Advanced' and 'normalization'


Note: 

Label extraction will be performed at the beginning of the running program , building '../reStays' folder by breaking data by subjects.
After running the first time the folder exists for next running and will not run twice unless you remove ../generate_dataset/erStays folder.
 
Folder Structure: 
```
 sudo apt-get install tree
 
 tree /path/to/root
 ```



Requirements: 

- numpy==1.19.5, 
- pandas==1.2.4,
- scipy==1.6.2, 
- scikit-learn==0.24.2,
- matplotlib==3.3.4, 
- seaborn==0.11.1,  
- imbalanced-learn==0.8.0, 
- tensorboard==2.7.0, 'tensorflow-gpu==2.5.0',  'tensorflow==2.7.0',  
- tqdm==4.59.0, 
- xgboost==1.3.3,
- shap==0.39.0, 
- keras==2.7.0, 


 - Some important features of next releases:
   - Advanced resampling methods for handling imbalanced data
   - Dynamic feature selection for chartevents items
   - Advanced outlier healing method
   - Advanced feature selection methods for algorithms
   - Tuning module for algorithms
   - Separated interpratation module


### Steps to Run:

**Step 0.** The app will run in two phases 
1. Label Extraction 
2. Preprocessing and Model building and training.
So run main.py to build '../erStays' folder by breaking stays by subjects and extract labels.
After extracting labels and building labeled data run configgen.ipynb and and set parameters and run main.py again to continue based on config.yml.
```
run ../generate_dataset/main.py

python generate_datasets/main.py   
```
Data generation and training will be notified through running based on config setting.

**Step 1.** Start with following config and run algorithms on "atAdmit" subtask, in this case only static and demographic features will be used :
```
dict_file = [{'data_dir': '../data'},{'subtask':{'isEarly' : 0, 'atDischarge':0, 'atAdmit':1}},
            {'feat_set' : {'static_feats':1, 'chart_events_feats':0, 'labevents_feats':0}}, 
            {'wt_forrolling':{'erwt_for_rolling':{12:0, 6:0, 2:0}, 'dischwt_for_rolling': {24:0, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'byAnomalyScore':0, 'normalization':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```

**Step 2.** Continue with "atDischarge" and "isEarly" subtasks one by one with only chartevents features :
```
dict_file = [{'data_dir': '../data'},{'subtask':{'isEarly' : 0, 'atDischarge':1, 'atAdmit':0}},
            {'feat_set' : {'static_feats':1, 'chart_events_feats':1, 'labevents_feats':0}}, 
            {'wt_forrolling':{'erwt_for_rolling':{12:1, 6:0, 2:0}, 'dischwt_for_rolling': {24:1, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'byAnomalyScore':1, 'normalization':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```
```         
dict_file = [{'data_dir': '../data'},{'subtask':{'isEarly' : 1, 'atDischarge':0, 'atAdmit':0}},
            {'feat_set' : {'static_feats':1, 'chart_events_feats':1, 'labevents_feats':0}}, 
            {'wt_forrolling':{'erwt_for_rolling':{12:1, 6:0, 2:0}, 'dischwt_for_rolling': {24:1, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'byAnomalyScore':1, 'normalization':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```

**Step 3.** Run step 2 and add labevents features (takes more run time):
```
dict_file = [{'data_dir': '../data'},{'subtask':{'isEarly' : 0, 'atDischarge':1, 'atAdmit':0}},
            {'feat_set' : {'static_feats':1, 'chart_events_feats':1, 'labevents_feats':1}}, 
            {'wt_forrolling':{'erwt_for_rolling':{12:1, 6:0, 2:0}, 'dischwt_for_rolling': {24:1, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'byAnomalyScore':1, 'normalization':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```
```
dict_file = [{'data_dir': '../data'},{'subtask':{'isEarly' : 1, 'atDischarge':0, 'atAdmit':0}},
            {'feat_set' : {'static_feats':1, 'chart_events_feats':1, 'labevents_feats':1}}, 
            {'wt_forrolling':{'erwt_for_rolling':{12:1, 6:0, 2:0}, 'dischwt_for_rolling': {24:1, 48:0}}}, 
            {'cat_encoding':{'le':1, 'woe':0, 'oh':0}}, 
            {'outlier_heal':{'byAnomalyScore':1, 'normalization':0, 'Advanced':0}}, 
            {'imp_type':{'mean':1, 'median':0,'mice':0 }}, 
            {'model':{'XGB':1, 'Logisticregression':1, 'MLP':1}}, 
            {'target':{'30':1, '7':1, 'ntw':1}}, 
            {'measure':{'f1_score':1, 'recall':0, 'accuracy':0 }},
            {'GPU':1}]
```

During above steps if GPU is not avaialable,  set GPU:0 (XGB runs only on GPU, recommended)


Note for running algorithms: 

    - XBG handles missing data automatically and is less sensitive to outliers
    - Imputing scaling before running linear algorithms we used is mandatory
    - In this version we used statis parameters for all input data, just tune the target weights for handling 
    imbalanced data. To get better results tuning module is recommended
    
    
