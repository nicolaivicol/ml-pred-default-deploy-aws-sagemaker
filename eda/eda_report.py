# %%
'''# **Exploratory Data Analysis**'''

# %%
'''
<a id="top"></a> <br>
**TOC**   
- [Params, Libs, Options](#params_libs_options)   
- [Load data](#load_data)  
- [Target: Default](#target)  
- [Descriptive summary](#descriptive_summary)  
    - [Numeric columns](#numeric_columns)    
    - [Categorical columns](#categorical_columns)    
- [Plot default rate by main categorical features](#plot_by_cat_feats)
- [Plot default rate by main numeric features](#plot_by_num_feats)
'''

# %%
'''
<a id="params_libs_options"></a> 
## Params, Libs, Options
'''

# %%
# load libs
import os
import numpy as np
# from pathlib import Path
# import plotly.graph_objects as go
import plotly.offline as py
# from plotly.subplots import make_subplots
from IPython.display import display

# cd at project's root folder to load modules
DIR_PROJ = 'klrn'
if DIR_PROJ in os.getcwd().split('/'):
    while os.getcwd().split(os.path.sep)[-1] != DIR_PROJ:
        os.chdir('..')
    DIR_PROJ = os.getcwd()
    print(f"working dir: {os.getcwd()}")
else:
    raise NotADirectoryError(f"please set working directory at project's root: '{DIR_PROJ}'")
# DIR_PROJ = (Path(__file__) / '..' / '..').resolve()

# load project modules
from config import MAP_DATA_COLS_TYPES, TARGET_NAME
from eda.utils_eda import (
    describe_numeric,
    display_descr_cat_freq,
    set_display_options,
    plot_agg_target_by_feat_env,
)
from etl import load_raw_data

# options
set_display_options()

# %%
'''
<a id="load_data"></a> 
## Load data
**Load data from disk**  
'''

# %%
df = load_raw_data()

# %%
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
display(df.head(5))

# %%
''' 
<a id="target"></a>  
## Target 
The target is binary telling whether a client has defaulted (default=1) or not (default=0). 
The default rate on the provided sample is about **1.43%**. That is a highly imbalanced dataset.
There are **10,000** NA values in the target column 'default' (**10%** out the total sample of 99,976 rows). 
These are the values we need to predict (test).
'''

# %%
target_is_na = df[TARGET_NAME].isna()
print(f"Default rate: {round(df.loc[~target_is_na, TARGET_NAME].mean()*100, 2)}%")
print(f"Percentage of NA values (to be predicted): {round(np.mean(target_is_na)*100, 2)}%")

# %%
''' 
<a id="descriptive_summary"></a> 
## Descriptive summary
'''

# %%
''' 
<a id="numeric_columns"></a> 
### Numeric columns
'''

# %%
cols_numeric = [k for k, v in MAP_DATA_COLS_TYPES.items() if v == 'numeric']
display(describe_numeric(df, cols_numeric))

# %%
''' 
<a id="categorical_columns"></a>  
### Categorical columns 
'''

# %%
cols_categorical = [k for k, v in MAP_DATA_COLS_TYPES.items() if v in ['categorical', 'boolean']]
display(display_descr_cat_freq(df, cols_categorical))

# %%
''' 
<a id="plot_by_cat_feats"></a>  
## Plot default rate by main categorical features
'''

# %%
cols = [
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
for col in cols:
    fig = plot_agg_target_by_feat_env(df, col, 'cat', min_size=0)
    py.iplot(fig)

# %%
''' 
<a id="plot_by_num_feats"></a>  
## Plot default rate by main numeric features
'''

# %%
cols = [
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
for col in cols:
    fig = plot_agg_target_by_feat_env(df, col, 'num', min_size=0)
    py.iplot(fig)
    # py.plot(fig)

# %%
'''
[top](#top)
## END OF REPORT
'''

# %%
'''
**Convert this .py script to .ipynb:**   
ipynb-py-convert eda/eda_report.py eda/eda_report.ipynb    

**Start python notebook:**      
cd /Users/nicolaiv/work/misc/klrn    
conda activate klrn-hw   
jupyter notebook  
'''
