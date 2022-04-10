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
- [Plot default rate by main numeric features](#plot_by_num_feats)
- [Plot default rate by main categorical features](#plot_by_num_feats)
- [XXX](#xxx)
'''

# %%
'''
<a id="params_libs_options"></a> 
## Params, Libs, Options
'''

# %%
# load libs
import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import plotly.graph_objects as go
import plotly.offline as py
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import lightgbm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
from eda.utils import describe_numeric, display_descr_cat_freq, set_display_options, plot_agg_target_by_feat_env
from config import MAP_DATA_TYPES

# options
set_display_options()

# %%
'''
<a id="load_data"></a> 
## Load data
**Load data from disk**  
'''

# %%
df = pd.read_csv(f'{DIR_PROJ}/data/dataset.csv', sep=';', na_values=['NA'])

# %%
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
display(df.head(5))

# %%
''' 
<a id="target"></a>  
## Target 
The target is binary telling whether a client has defaulted (default=1) or not (default=0). 
The default rate on the provided sample is about **1.43%**.
There are **10,000** NA values in the target column 'default' (**10%** out the total sample of 99,976 rows). 
These are the values we need to predict (test).
'''

# %%
target_is_na = df['default'].isna()
print(f"Default rate: {round(df.loc[~target_is_na, 'default'].mean()*100, 2)}%")
print(f"Percentage of NA values (to be predicted): {round(np.mean(target_is_na)*100, 2)}%")

# %%
''' 
<a id="descriptive_summary"></a> 
## Descriptive summary of all data 
'''

# %%
''' 
<a id="numeric_columns"></a> 
### Numeric columns
'''

# %%
cols_numeric = [k for k, v in MAP_DATA_TYPES.items() if v == 'numeric']
display(describe_numeric(df, cols_numeric))

# %%
''' 
<a id="categorical_columns"></a>  
### Categorical columns 
'''

# %%
cols_categorical = [k for k, v in MAP_DATA_TYPES.items() if v in ['categorical', 'boolean']]
display(display_descr_cat_freq(df, cols_categorical))


# %%
''' 
<a id="plot_by_cat_feats"></a>  
## Plot default rate by main categorical features
'''

# %%
cols = ['account_status', 'merchant_category', 'has_paid']
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
    'account_amount_added_12_24m',
    'account_incoming_debt_vs_paid_0_24m',
    'age',
    'avg_payment_span_0_12m',
    'num_active_inv',
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
jupyter notebook --notebook-dir=work/git/nicv-onboard-churn/   
'''

