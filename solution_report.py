# %%
'''
<b><font size="+3">Solution report: Klarna Interview Homework</font></b><br><br>
<font size="+1">Nicolai Vicol (<a href="mailto:nicolaivicol@gmail.com">nicolaivicol@gmail.com</a>)</font><br>
<font size="+1">2022-04-15</font>
'''

# %%
'''
<a id="top"></a><br>
**Contents**   
- [TLDR](#tldr)
- [Data](#load_data)  
- [Target Variable: default](#target)  
    - [Obs 1: Imbalanced Data](#imbalanced_data)
    - [Obs 2: Train/Evalution Split](#train_test_split)
- [Descriptive Summary](#descriptive_summary)  
    - [Numeric Columns](#numeric_columns)    
    - [Categorical Columns](#categorical_columns)    
- [Plot default rate by main categorical features](#plot_by_cat_feats)
- [Plot default rate by main numeric features](#plot_by_num_feats)
- [Features Engineering](#features_engineering)
- [Features Selection](#features_selection)
- [Hyper-parameters Tuning](#hyper_parameters_tuning)
- [Model performance](#model_performance)
- [Pipeline: steps & instructions](#)
'''

# %%
# load libs
import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'  # to suppress a warning by numexpr
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
# from plotly.subplots import make_subplots
from IPython.display import display

# cd at project's root folder to load modules
DIR_PROJ = 'klrn'
if DIR_PROJ in os.getcwd().split('/'):
    while os.getcwd().split(os.path.sep)[-1] != DIR_PROJ:
        os.chdir('')
    DIR_PROJ = os.getcwd()
    # print(f"working dir: {os.getcwd()}")
else:
    raise NotADirectoryError(f"please set working directory at project's root: '{DIR_PROJ}'")

# load project modules
import config
from utils import (
    describe_numeric,
    display_descr_cat_freq,
    set_display_options,
    plot_many_agg_target_by_feat,
)
from etl import load_raw_data

# options
set_display_options()

# %%
'''
<a id="tldr"></a> 
## TLDR
* sss
* sdd
'''

# %%
'''
<a id="load_data"></a> 
## Data
Data provided in the file *dataset.csv* contains **99976** rows and **43** columns.    
Sample of data:
'''

# %%
df = load_raw_data()
display(df.head(5))
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

# %%
''' 
<a id="target"></a>  
## Target variable: default
The target is binary telling whether a client has defaulted (default=1) or not (default=0). 
'''

# %%
_ = display_descr_cat_freq(df, [config.TARGET_NAME])

# %%
'''
<a id="imbalanced_data"></a>  
### Obs 1: Imbalanced Data
The default rate on the provided sample is about **1.43%**.    
That is a highly imbalanced dataset.    
It will be a good idea to use a stratified cross validation.
'''

# %%
target_is_na = df[config.TARGET_NAME].isna()
print(f"Default rate: {round(df.loc[~target_is_na, config.TARGET_NAME].mean()*100, 2)}%")

# %%
'''
### Obs 2: Train/Evaluation Split
There are **10,000** NA values in the target column 'default' (**10%** out the total sample of 99,976 rows).    
These are the values the candidates needs to predict.    
This prediction will be used by Klarna interviewers to evaluate the model on unseen data.      
The train and evaluate data are disjoint sets by uuid (user ID) with a proportion of **90%/10%** respectively (89,976 / 10,000 users).    
The evaluation set was drawn randomly by uuid.    
Hence, the cross-validation should also be performed on folds drawn randomly by uuid and train/validate splits seprated by uuid.    
'''

# %%
print(f"Percentage of NA values (to be predicted): {round(np.mean(target_is_na)*100, 2)}%")
n = len(set(df.loc[target_is_na, 'uuid']).intersection(set(df.loc[~target_is_na, 'uuid'])))
print(f'Number of uuid values present in both Train & Evaluate: {n}')

# %%
''' 
<a id="descriptive_summary"></a> 
## Descriptive Summary

<a id="numeric_columns"></a> 
### Numeric Columns
Observations:
* Numeric features are of several types: 
    * counting actions/statuses, usually starting with prefix `num_`, e.g. `num_arch_dc_0_12m`, `num_unpaid_bills`
    * monetary sums/amounts, e.g. `max_paid_inv_0_12m`, `sum_capital_paid_account_0_12m`
    * time spans, e.g. `account_days_in_dc_12_24m`, `avg_payment_span_0_12m`
    * about user, `age`
* All numeric features have non-negative values.
* Some features have NA values (e.g. `account_days_in_dc_12_24m`, `account_incoming_debt_vs_paid_0_24m`).
* For a GBM model, replacing NA values with -1 is reasonable. 
'''

# %%
cols_numeric = [k for k, v in config.MAP_DATA_COLS_TYPES.items() if v == 'numeric']
display(describe_numeric(df, cols_numeric))

# %%
''' 
<a id="categorical_columns"></a>  
### Categorical Columns
Observations:
* Categorical features are of 3 types:
    * with integer values, non-negative, mapping a status and having low cardinality (up to 5), e.g. `account_status`
    * with text values, describing categories and having higher cardinality (from 8 to 57 unique values), e.g. `merchant_group`, `name_in_email`
    * boolean, e.g. `has_paid`
* Some categorical features have missing values. These happens only to integer-like features, having a minimum value of 1. Replacing NA with 0, as a new category, is reasonable.
'''

# %%
cols_categorical = [k for k, v in config.MAP_DATA_COLS_TYPES.items() if v in ['categorical', 'boolean'] and k != config.TARGET_NAME]
_ = display_descr_cat_freq(df, cols_categorical)

# %%
''' 
<a id="plot_by_cat_feats"></a>  
## Plot default rate by main categorical features
'''

# %%
names_feat = [
    'account_worst_status_0_3m',
    'status_last_archived_0_24m',
    'status_max_archived_0_24_months',
    'status_2nd_last_archived_0_24m',
    'status_max_archived_0_12_months',
    'account_status',
    'account_worst_status_6_12m',
    'status_3rd_last_archived_0_24m',
    'has_paid',
    'merchant_group',
    'account_worst_status_3_6m',
    'worst_status_active_inv',
]
fig = plot_many_agg_target_by_feat(df, names_feat, type_feat='cat', min_size=0, y2_range=[-0.02, 0.25])
py.iplot(fig)

# %%
''' 
<a id="plot_by_num_feats"></a>  
## Plot default rate by main numeric features
'''

# %%
names_feat = [
    'num_arch_ok_0_12m',
    'num_arch_ok_12_24m',
    'num_unpaid_bills',
    'avg_payment_span_0_12m',
    'num_active_div_by_paid_inv_0_12m',
    'avg_payment_span_0_3m',
    'num_active_inv',
    'age',
    'recovery_debt',
    'num_arch_dc_0_12m',
    'max_paid_inv_0_24m',
    'num_arch_dc_12_24m',
]
fig = plot_many_agg_target_by_feat(df, names_feat, min_size=0, y2_range=[-0.02, 0.25])
py.iplot(fig)

# %%
'''
<a id="choice"></a>  
## Choice of Model & Metric
I have decided to use a GBM model for this task.     
That's because this type of model deals well with nonlinearities and interactions 
between features and requires minimal or no feature engineering.    
Also these models perform well on tabular data and on volumes of data in the order of 100K rows.   
Among the usual cadidates: LightGBM, XGBoost and CatBoost, I have decide to use the **XGBoost** library.     
That's because it integrates better with AWS and I want to deploy it later to AWS SageMaker.

I have decide to use the **AUC** metric (describing the ROC curve) for this classification task.    
This metric suits better the case of imbalanced data.
'''

# %%
'''
<a id="features_engineering"></a>  
## Features Engineering

The GBM model (XGBoost) that is used in this solution, deals well with raw features, with their nonlinearities and interactions.     
The model is also scale invariant to features. In other words, the features engineering can be minimal.     
The only requiremnt is to have a dense matrix, i.e. replace NA values with non-NA values.    

In the end, only the following transformation were performed to the raw features:
* replace NA values with 0 in the categorical features (those having integer values)
* text categorical features were encoded using a simple numeric (ordinal) encoding, but only the first 10 most frequent were kept, all other values fall under a common miscellaneous category.
* replace NA values with -1 in numeric features
* convert boolean columns to numeric

No new features were added.
'''

# %%
'''
<a id="features_selection"></a>
## Features Selection
Features selection was performed with [boruta](https://mbq.github.io/Boruta/) library.     
Out of **41** initial features, **26** were selected and **15** were dropped.     
Result of boruta selection:
'''

# %%
df_res_feat_slct = pd.read_csv(config.FILE_SLCT_FEATS_RES_BORUTA)
display(df_res_feat_slct)

# %%
'''
The features importance collected from many iterations during CV are illustrated below.
'''

# %%
feat_imp = pd.read_csv(config.FILE_CV_FEAT_IMP)
feat_imp = feat_imp.groupby(
    by='feature', as_index=True
).agg(
    mean=('importance', np.mean),
    sd=('importance', np.std),
    count_folds=('importance', np.size)
).reset_index()
feat_imp.sort_values(by='mean', axis=0, ascending=False, inplace=True, na_position='last')

fig = go.Figure()
feat_imp = feat_imp.head(50)
feat_imp.sort_values(by=['mean'], ascending=True, inplace=True)
fig.add_trace(
    go.Bar(
        x=np.round(feat_imp['mean'], 3),
        y=feat_imp['feature'],
        marker=dict(color="grey", opacity=0.6),
        orientation='h',
        error_x=dict(
            type="data",
            array=np.round(feat_imp['sd'], 3),
            arrayminus=np.round(feat_imp['sd'], 3),
            visible=True
        ),
        name='Importance')
)
fig.update_layout(
    title='Features Importance',
    xaxis=dict(title='Importance', range=[0, max(feat_imp['mean'] + 0.02)]),
    autosize=True,
    height=800,
)
py.iplot(fig)

# %%
'''
<a id="hyper_parameters_tuning"></a>
## Hyper-parameters Tuning
The search for best hyper-parameters was done via a randomized grid search using stratified k-fold validation.    
For a given combination of parameters a cross-validation of k=4 folds is performed (k is small for faster iterations).
The algorithm was allowed to run for **200** random combinations of parameters out of 2,916 possible from the following grid:
'''

# %%
display(config.TUNE_PARAMS_GRID)

# %%
'''
The best combination found is the following:
'''

# %%
with open(config.FILE_TUNE_PARAMS_BEST, 'r') as f:
    best_pars = json.load(f)
display(best_pars)

# %%
'''
Top 20 combinations of parameters:
'''

# %%
df_res_tune = pd.read_csv(config.FILE_TUNE_ALL_PARAMS_COMBS)
cols_show = ['mean_test_score', 'std_test_score'] + [f'param_{p}' for p in config.TUNE_PARAMS_GRID.keys()]
df_res_tune = df_res_tune[cols_show]
df_res_tune.columns = [f.replace('param_', '') for f in df_res_tune.columns]
display(df_res_tune.head(20))

# %%
'''
All 200 combinations attempted in a parallel coordinates plot:
'''

# %%
fig = px.parallel_coordinates(
    df_res_tune,
    color='mean_test_score',
    dimensions=['mean_test_score'] + list(config.TUNE_PARAMS_GRID.keys()),
    color_continuous_scale=['red', 'yellow', 'green'],
    color_continuous_midpoint=np.median(df_res_tune['mean_test_score']),
)
py.iplot(fig)

# %%
'''
<a id="model_performance"></a>
## Model performance
A stratified k-fold cross-validation was performed using the best hyper-parameters.    
This was necessary to have a better estimate of out-of-sample performance of the model.    
In contrast to the CV iterations for hyper-parameters which used k=4 folds, this time we used k=16 folds.
* train: avg AUC = **0.938** (std ROC_AUC = 0.001)
* valid: avg AUC = **0.912** (std ROC_AUC = 0.013)
* test: ROC_AUC = **0.905**

The AUC on test data is close (within 1 std dev) to the average AUC on CV folds, meaning that the model can be trusted.     
The test data is a portion of data of 30% of the initial data, which was not involved in hyper-parameters tuning.    
The final model used for inference is trained on 100% of data.
'''

# %%
df_cv_metrics = pd.read_csv(config.FILE_CV_METRICS)
df_test_metrics = pd.read_csv(config.FILE_TEST_METRICS)
test_score = np.mean(df_test_metrics['test_score'])
avg_train, std_train = np.mean(df_cv_metrics['train_score']), np.std(df_cv_metrics['train_score'])
avg_test, std_test = np.mean(df_cv_metrics['test_score']), np.std(df_cv_metrics['test_score'])
_ = f'Metrics: \n' \
    f' - train: avg AUC = {avg_train:.3f} (std AUC = {std_train:.3f})\n' \
    f' - valid: avg AUC = {avg_test:.3f} (std AUC = {std_test:.3f})\n' \
    f' - test: AUC = {test_score:.3f}'
print(_)


# %%
'''
<a id="tr"></a>
## Model performance
'''


# %%
'''
<a id="workflow_step_instructions"></a>
## Pipeline: steps & instructions
### Prerequisites
* create conda environment in order to be able to run the code in this project
    * `cd klrn` (change directory to where the project is located)
    * `conda env create -f conda.yml`
    * `conda activate klrn-hw`

### Configs
The configuration of the entire pipeline is stored in `config.py`.          
This config file stores all the parameters regarding:      
data & artifacts file paths, each step of the pipeline, like feat selection, cv, etc.

### ETL & Features Engineering
Module `etl.py` has a series of functions that are used when loading & preparing data for training and CV.   
Data is loaded and transformed at runtime and no transformed data is saved during this pipeline.    

### Features selection with Boruta
Run script `select_feats.py` or `select_feats()` from it.     
ETA < 10 min    

### Tune hyper-parameters via CV
Run script `tune_hyper_params.py` or `tune_hyper_params()` from it.     
ETA = 20-30 min (long because it runs train & predict many times: 200 params combination x 4 folds)  

### Estimate model performance via CV
Run script `cv.py` or `cv()` from it.    
ETA < 5 min (fast because it runs k=16 folds only for one combination of hyper-prameters, the best found in tuning)

### Train model
Run script `train.py` or `train()` from it.      
The trained model is saved as a file to `/artifacts`.     
Later this will be loaded for prediction.

### Predict
Run script `predict.py` or `predict()` from it.

### Run pipeline entirely
Run script `run_pipeline.py` or `run_pipeline()` from it.

### Generate solution report as a self-contained html file
* Change directory to where the project is located:
    * `cd /Users/nicolaiv/work/misc/klrn`
* Activate conda environment:
    * conda activate klrn-hw
* Convert .py file to .ipynb notebook:
    *`ipynb-py-convert solution_report.py artifacts/solution_report.ipynb`
* Start jupyter:
    *`jupyter notebook`
* Run all cells:
    * jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --inplace --execute artifacts/solution_report.ipynb
* Download as html with TOC and hidden code:
    * `jupyter nbconvert artifacts/solution_report.ipynb --to=html_toc --TemplateExporter.exclude_input=True`
'''

# %%
'''
[Go to top](#top)    
END OF REPORT
'''
