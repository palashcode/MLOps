from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp
import config


olist_pipe = Pipeline(
    [
        ('fill_missing_categorical',
            pp.FillMissingCategorical(variables=config.CATEGORICAL_VARS_WITH_NA)),
         
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
         
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
         
        ('Linear_model', Lasso(alpha=0.005, random_state=0))
    ]
)