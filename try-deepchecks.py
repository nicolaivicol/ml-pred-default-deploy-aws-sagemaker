import logging
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

import config
from etl import load_train_test_data


logger = logging.getLogger('deepchecks')

logger.info('Load data')
X_train, X_test, y_train, y_test = load_train_test_data(selected_feats_only=True)
label_col = 'target'

df_train = X_train.copy()
df_train['target'] = y_train

df_test = X_test.copy()
df_test['target'] = y_test

ds_train = Dataset(df_train, label=label_col, cat_features=[])
ds_test = Dataset(df_test, label=label_col, cat_features=[])

logger.info('load model')
params_model = config.get_params_model()
model = XGBClassifier(**params_model)
model.fit(X_train, y_train)
y_pred_prob_test = model.predict_proba(X_test)[:, 1]
score_test = roc_auc_score(y_test, y_pred_prob_test)

logger.info('run deepchecks suite')
suite = full_suite()
res = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)
res.save_as_html('artifacts/deepchecks.html')
