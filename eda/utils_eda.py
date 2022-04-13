import numpy as np
import pandas as pd
from IPython.display import display, HTML
import plotly.graph_objects as go
import warnings


HEIGHT_PLOT = 650


def describe_numeric(df, cols_num=None, percentiles=None):
    """
    Describe numeric columns
    :param df: pandas data frame
    :param cols_num: numeric columns to describe, by default: identified automatically
    :param percentiles: percentiles to compute, default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :return: pandas df with stats
    """
    if cols_num is None:
        cols_num = list(df.head(1).select_dtypes(include=['number']).columns)
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if len(cols_num) == 0:
        return None
    d_describe = df[cols_num].describe(percentiles=percentiles).T
    d_describe['count_nan'] = df.isnull().sum()
    d_describe['prc_nan'] = 1 - d_describe['count'] / float(df.shape[0])
    return d_describe


def describe_categorical(df, cols=None):
    """
    Describe categorical columns
    :param df: pandas data frame
    :param cols: categorical columns to describe, by default: identified automatically
    :return: pandas df with stats
    """
    if cols is None:
        cols = list(df.head(1).select_dtypes(include=['object']).columns)
    if len(cols) == 0:
        return None
    d_describe = df[cols].astype('category').describe().T
    return d_describe


def describe_categorical_freq(x: pd.Series, name: str = None, max_show: int = 10, min_prc: float = 0.001):
    """
    Describe series with categorical values (counts, frequency)
    :param x: series to describe
    :param name: name
    :param max_show: max values to show
    :param min_prc: minimum size (in %) for the category to show in stats
    :return: pandas df with stats
    """
    if name is None:
        try:
            name = x.name
        except:
            name = 'value'
    tmp = pd.DataFrame({name: x})

    agg = tmp.groupby([name], dropna=False, as_index=True).agg({name: len}).rename(columns={name: 'count'})
    agg['percentage'] = agg['count'] / sum(agg['count'])
    agg.sort_values(['count'], ascending=False, inplace=True)
    agg.reset_index(drop=False, inplace=True)
    filter_out = (((agg['percentage'] < min_prc)
                   & (pd.Series(range(len(agg))) > max_show))
                  | (pd.Series(range(len(agg))) > max_show))
    agg = agg.loc[~filter_out, ]
    return agg


def display_descr_cat_freq(df, cols=None, skip_freq_cols=None):
    """
    Describe categorical columns in dataframe (counts, frequency)
    :param df: data frame
    :param cols: for which columns to compute statistics, by default: identifed automatically
    :param skip_freq_cols: which columns to skip
    :return: pandas df with stats
    """
    if cols is None:
        cols = list(df.head(1).select_dtypes(include=['object']).columns)
    if skip_freq_cols is None:
        skip_freq_cols = []
    if len(cols) == 0:
        return None
    display(describe_categorical(df, cols))
    for col in cols:
        if col not in skip_freq_cols:
            display(HTML(f'<br><b>{col}</b>'), describe_categorical_freq(df[col]))


def set_display_options():
    """
    Set display options for numbers, table width, etc.
    :return: None
    """
    pd.set_option('plotting.backend', 'plotly')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('max_colwidth', 150)
    pd.set_option('display.precision', 2)
    pd.set_option('display.chop_threshold', 1e-6)
    # pd.set_option('expand_frame_repr', True)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    warnings.simplefilter('ignore')
    display(HTML("<style>.container { width:80% !important; }</style>"))


def agg_target_by_numeric_feat(target, feat, bins=10, bin_by='range', bin_min_size=0.0, include_nan=True):
    """
    Aggregate a target variable by a numeric feature variable
    :param target: target values as list/array/series
    :param feat: numeric feature values as list/array/series
    :param bins: how many bins? Default: 10
    :param bin_by: how to form bins? 'quantile': by equal quantiles, 'range': by equal ranges (as for histograms)
    :param bin_min_size: Minimum size (as percent) of bin to be included in table. Default: 0.0 (all included)
    :param include_nan: Include nan as bin in aggregation? Default: True (nan will eventually appear in the first row)
    :return: data frame with stats
    """

    d_tmp = pd.DataFrame()
    d_tmp['target'] = target
    d_tmp['feat'] = feat

    if bin_by == 'range':
        intervals = pd.cut(x=d_tmp['feat'], bins=bins)
    elif bin_by == 'quantile':
        intervals = pd.qcut(d_tmp['feat'], q=bins)
    else:
        raise ValueError("allowed values for arg 'bin_by': 'range', 'percentile'.")

    d_tmp['bin_left'] = intervals.apply(lambda x: x.left)
    d_tmp['bin_mid'] = intervals.apply(lambda x: x.mid)
    d_tmp['bin_right'] = intervals.apply(lambda x: x.right)

    agg_cols = ['bin_left', 'bin_mid', 'bin_right']
    if include_nan:
        for col in agg_cols:
            d_tmp[col] = d_tmp[col].astype('str')

    d_agg = d_tmp.groupby(
        by=agg_cols, as_index=True
    ).agg(
        mean=('target', np.mean),
        count=('target', np.size)
    ).reset_index()
    d_agg['count_prc'] = d_agg['count'] / d_tmp.shape[0]

    if 0 <= bin_min_size <= 1:
        d_agg = d_agg[d_agg['count_prc'] > bin_min_size]
    else:
        raise ValueError("arg 'bin_min_size': 0 < 'bin_min_size' < 1")

    if include_nan:
        for col in agg_cols:
            d_agg[col] = d_agg[col].astype('float')

    d_agg.sort_values(by='bin_mid', axis=0, ascending=True, inplace=True, na_position='first')
    return d_agg


def agg_target_by_cat_feat(target, feat, cat_min_size=0.0, include_nan=True):
    """
    Aggregate the target variable by a categorical feature variable
    :param target: target values as list/array/series
    :param feat: categorical feature values as list/array/series
    :param cat_min_size: Minimum size (as percent) of category to be included in table. Default: 0.0 (all included)
    :param include_nan: Include nan as category in aggregation? Default: True (a new category 'nan' will be introduced)
    :return: data frame with stats
    """

    d_tmp = pd.DataFrame()
    d_tmp['target'] = target
    d_tmp['feat'] = feat

    if include_nan:
        d_tmp['feat'] = d_tmp['feat'].astype('str')
        d_tmp.loc[d_tmp['feat'].isnull(), 'feat'] = 'nan'

    d_agg = d_tmp.groupby(
        by='feat', as_index=True
    ).agg(
        mean=('target', np.mean),
        count=('target', np.size)
    ).reset_index()
    d_agg['count_prc'] = d_agg['count'] / d_tmp.shape[0]

    if 0 <= cat_min_size <= 1:
        d_agg = d_agg[d_agg['count_prc'] > cat_min_size]
    else:
        raise ValueError("arg 'cat_min_size': 0 <= 'cat_min_size' <= 1")

    d_agg.sort_values(by='count', axis=0, ascending=False, inplace=True, na_position='first')
    return d_agg


def plot_agg_target_by_feat(y_agg, x_labels, bins_size=None, y_agg_full=None,
                            name_target='Target', name_agg_func='avg', name_labels='Feature',
                            name_bins='bin', y2_range=None, is_first_nan=False):
    """
    Plot aggregated target variable by a feature variable (numeric or categorical)
    :param y_agg: Aggregated value of target by bin/category.
    :param x_labels: Mid of bin intervals or Categories
    :param bins_size: Size of bins/categories as percent (0 to 1) from histogram
    :param y_agg_full: Aggregated value of target on entire population
    :param name_target: Default: 'Target'
    :param name_agg_func: Name of aggregation function, Default: 'avg'
    :param name_labels: Name of labels series, Default: 'Feature'
    :param name_bins: Hiw data is binned, usually: 'cat' for categorical  or 'bin' for numeric
    :param y2_range: Range of target, default: None (automatically set)
    :param is_first_nan: Is first bar for nan values? Default: False
    :return: plotly object
    """

    fig = go.Figure()
    # add histogram bars on y-axis on the left
    if bins_size is None:
        bins_size = [np.nan] * len(x_labels)
    fig.add_trace(
        go.Bar(x=x_labels,
               y=np.round(bins_size, 3),
               name='histogram',
               opacity=0.4,
               marker_color='grey',
               yaxis='y')
    )
    # add annotation on top of first bar to indicate that this bar is for NaN values
    if is_first_nan:
        fig.add_annotation(x=x_labels[0], y=bins_size[0], text="NaN")
        fig.update_annotations(dict(
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=5,
            ax=0,
            ay=-40
        ))
    # add aggregated value of target variable per bin/category as red dot markers
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=np.round(y_agg, 3),
            mode='markers',
            opacity=0.7,
            name=f'{name_agg_func} per {name_bins} (RHS)',
            marker=dict(
                color='red',
                size=15,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            yaxis='y2')
    )
    # add the aggregated value of target value over the whole population as red line
    if y_agg_full is not None:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[np.round(y_agg_full, 3)] * len(x_labels),
                mode='lines',
                name=f'{name_agg_func} per all (RHS)',
                opacity=0.7,
                line=dict(color='red',
                          width=2),
                yaxis='y2')
        )
    # titles, x-axis and y-axis names
    fig.update_layout(
        title=f'<b>{name_target}</b> by <b>{name_labels}</b>',
        xaxis=dict(title=f'<b>{name_labels}</b>'),
        yaxis=dict(title='Histogram',
                   range=[-0.02, 1.02],
                   showgrid=False),
        yaxis2=dict(title=f'<b>{name_target} {name_agg_func}</b>',
                    side='right',
                    overlaying='y',
                    range=y2_range),
        height=HEIGHT_PLOT,
        margin=dict(l=50, r=25, t=25, b=50),
        autosize=True,
    )
    return fig


def plot_agg_target_by_feat_env(d, name_feat, type_feat='num', target_name='default',
                                y2_range=None, min_size=0.0025, show_legend=True):
    """
    Envelope on function plot_agg_target_by_feat
    :param d: data frame
    :param name_feat: name of feature
    :param type_feat: 'num'/'numerical' or 'cat'/'categorical'
    :param target_name: default: 'default'
    :param y2_range: default: [-0.02, 1.02]
    :param min_size: minimum size of bin/category to be shown on plot (percent), default: 0.0025
    :param show_legend: default: True
    :return: plotly object
    """

    if y2_range is None:
        y2_range = [-0.02, 1.02]

    is_first_nan = False
    if type_feat in ['cat', 'categorical']:
        col_x_labels, name_bins = 'feat', 'cat'
        d_agg = agg_target_by_cat_feat(
            target=d[target_name],
            feat=d[name_feat],
            cat_min_size=min_size,
        )
    elif type_feat in ['num', 'numeric']:
        col_x_labels, name_bins = 'bin_mid', 'bin'
        d_agg = agg_target_by_numeric_feat(
            target=d[target_name],
            feat=d[name_feat],
            bins=25,
            bin_by='range',
            bin_min_size=min_size,
        )
        # susbtitute nan value with a fake value on the far left of the lowest value
        which_is_nan = d_agg['bin_mid'].isnull()
        is_first_nan = any(which_is_nan)
        if is_first_nan:
            fake_value_for_nan = np.min(d_agg['bin_mid']) - 3 * (
                    np.max(d_agg['bin_mid']) - np.min(d_agg['bin_mid'])) / 25
            d_agg.loc[which_is_nan, 'bin_mid'] = fake_value_for_nan
    else:
        raise ValueError("allowed values for type_feat: 'cat'/'categorical', 'num'/'numerical'")

    fig = plot_agg_target_by_feat(
        y_agg=d_agg['mean'].values,
        x_labels=d_agg[col_x_labels].values,
        bins_size=d_agg['count_prc'].values,
        y_agg_full=np.mean(d[target_name]),
        name_target=target_name.capitalize(),
        name_agg_func='ratio',
        name_labels=name_feat,
        name_bins=name_bins,
        y2_range=y2_range,
        is_first_nan=is_first_nan)

    fig.update_layout(showlegend=show_legend)
    return fig
