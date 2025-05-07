---
layout: wide_default
---

## Assignment 8 -- Predicting Home Sale Prices

Steps to this assignment. 

1. y-train is np.log(v_SalePrice), X_train is all of the predictors of home sale price
2. Preprocess the data 
3. Tune parameters of models and decide on optimal model
4. K-Folds to test/validate model
5. Create Pipeline which does all of this 
6. Then at the very very end, test on holdout. 


```python
## Batch Import Packages from ML_Worksheet 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from df_after_transform import df_after_transform
from sklearn import set_config
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegression, RidgeCV, LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
    TunedThresholdClassifierCV
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import LinearSVC


import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
```


```python
## Importing the dataset

housing = pd.read_csv('input_data2/housing_train.csv')
y = np.log(housing.v_SalePrice)
housing = housing.drop('v_SalePrice',axis=1) # so not to include it in the training dataset
```


```python
# splitting the dataset into training and testing sets
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng)

```

## Preprocessing Pipeline


```python
# create a preprocessing pipeline for the data
cat_cols = housing.select_dtypes(include=['object']).columns[1:].tolist()  # Exclude the first column which is 'parcel'

exclude = ['parcel'] # Exclude 'parcel' from numerical columns as it is not a feature but an identifier
cat_include = ['v_Overall_Qual', 'v_MS_SubClass'] # Include 'v_Overall_Qual' in categorical columns

number_cols = housing.select_dtypes(include=np.number).columns.tolist()  # Get all numerical columns
cat_cols = cat_cols + cat_include

numer_pipe = make_pipeline(SimpleImputer(), # blank fill with mean for numerical vars
                           StandardScaler()) # then scale it

cat_pipe   = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')) # fill in missing values with the most frequent value, then one-hot encode

# combine those pipes into "preprocess" pipe

number_cols = [col for col in number_cols if col not in exclude and col not in cat_cols]
cat_cols = [col for col in cat_cols if col not in exclude and col not in number_cols]

preproc_pipe = ColumnTransformer(  
    [ # arg 1 of ColumnTransformer is a list, so this starts the list
    # a tuple for the numerical vars: name, pipe, which vars to apply to
    ("num_impute", numer_pipe, number_cols), # Excludes the first column which is 'parcel',
    # a tuple for the categorical vars: name, pipe, which vars to apply to
        
    ("cat_trans", cat_pipe, cat_cols) # Excludes the first column which is 'parcel')
    # exclude from number pipe n--> number pipe exclusion pattern equals pattern = 
    ]
    , # ColumnTransformer can take other args, most important: "remainder"
    remainder = 'drop' # you either drop or passthrough any vars not modified above
)


```


```python
lasso_pipe = make_pipeline(
    preproc_pipe,
    LassoCV())

# I used "Pipeline" not "make_pipeline" bc I wanted to name the steps
lasso_pipe = Pipeline([('columntransformer',preproc_pipe),
                 ('feature_create','passthrough'), 
                 ('feature_select','passthrough'), 
                 ('reg', LassoCV()),
                ])

```

## Make Stacked Regressors

```python
# This stacks multiple regression models together, using the predictions of the first models as features for the final model.
# The final model, IN THIS PIECE OF CODE COPIED FORM SKLEARN DOCS, is a Gradient Boosting Regressor.

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
estimators = [('ridge', RidgeCV()),
              ('lasso', LassoCV(random_state=42)),
              ('knr', KNeighborsRegressor(n_neighbors=20,
                                          metric='euclidean'))]
estimators2  =[('ridge', RidgeCV()),
               #('lasso', LassoCV(random_state=42, tol= 1e-6)),
               ('knr', KNeighborsRegressor(metric='euclidean', n_neighbors = 25)), 
               ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
               ('hgbr', HistGradientBoostingRegressor())]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
final_estimator = GradientBoostingRegressor(
    n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
    random_state=42)
reg_boss = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator)

reg_boss2 = StackingRegressor(estimators=estimators2)

```

## Run Model Cross Validation to test on training set

```python
gridsearch = GridSearchCV(
    lasso_pipe,
    param_grid=[ # linear models with built in coef selection
                 {'reg': [Ridge(alpha=a) for a in np.arange(1,12,5)], 
                 'feature_create':['passthrough']}, #,PolynomialFeatures(interaction_only=True)]},
                
                # GBR doesntt like sparse input arrays 
                {'reg':[HistGradientBoostingRegressor(), reg_boss, reg_boss2],
                 'columntransformer__cat_trans__onehotencoder__sparse_output':[False]}
                
                # OLS but with feature selection
                #{'reg':[LinearRegression()],
                 ## for OLS, create interactition but then prune back...
                 #'feature_create': [PolynomialFeatures(interaction_only=True)],
                 #'feature_select': [SelectKBest(f_regression, k=i) for i in [10,30,50,70] ]}
                #
                ],
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=3
)
```


```python
gridsearch2 = GridSearchCV(
    lasso_pipe,
    param_grid=[ # linear models with built in coef selection
                 {'reg': [reg_boss2],
                  'reg__rf__n_estimators': [250],

                  'reg__knr__n_neighbors': [5,10],
                  'reg__knr__weights': ['distance'],
                  # 'reg__knr__metric': ['euclidean', 'manhattan'], euclidean is better
                  
                  'reg__hgbr__learning_rate': [0.1],

                # (optional) tune final estimator inside reg_boss2
                  # 'reg__final_estimator__learning_rate': [0.05, 0.1],
                  # 'reg__final_estimator__n_estimators': [10, 25, 50],
                  'columntransformer__cat_trans__onehotencoder__sparse_output':[False]}
                
                # OLS but with feature selection
                #{'reg':[LinearRegression()],
                 ## for OLS, create interactition but then prune back...
                 #'feature_create': [PolynomialFeatures(interaction_only=True)],
                 #'feature_select': [SelectKBest(f_regression, k=i) for i in [10,30,50,70] ]}
                #
                ],
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=3
)
```

## Train
```python
gridsearch.fit(X_train, y_train)
```


```python
# in the best estimator, access the coefs on the reg step's final estimator
# how much weight is applied to each input estimator?
gridsearch.best_estimator_.named_steps['reg'].final_estimator_.coef_
```




    array([0.30128563, 0.0303143 , 0.20974236, 0.49677478])




```python

print(f'Best parameters: {gridsearch.best_params_}')
print(f'Best cross-validated R^2 score: {gridsearch.best_score_:.4f}')

```

    Best parameters: {'columntransformer__cat_trans__onehotencoder__sparse_output': False, 'reg': StackingRegressor(estimators=[('ridge', RidgeCV()),
                                  ('knr',
                                   KNeighborsRegressor(metric='euclidean',
                                                       n_neighbors=25)),
                                  ('rf', RandomForestRegressor(random_state=42)),
                                  ('hgbr', HistGradientBoostingRegressor())])}
    Best cross-validated R^2 score: 0.9054



```python
# show which models were slowest to fit and train

pd.DataFrame(gridsearch.cv_results_)[['mean_fit_time','mean_score_time','mean_test_score','std_test_score','params']].sort_values(by='mean_fit_time', ascending=False)
```




```python
gridsearch2.fit(X_train, y_train)
```

```python
print(f'Best parameters: {gridsearch2.best_params_}')
print(f'Best cross-validated R^2 score: {gridsearch2.best_score_:.4f}')
```

    Best parameters: {'columntransformer__cat_trans__onehotencoder__sparse_output': False, 'reg': StackingRegressor(estimators=[('ridge', RidgeCV()),
                                  ('knr',
                                   KNeighborsRegressor(metric='euclidean',
                                                       n_neighbors=25)),
                                  ('rf', RandomForestRegressor(random_state=42)),
                                  ('hgbr', HistGradientBoostingRegressor())]), 'reg__hgbr__learning_rate': 0.1, 'reg__knr__n_neighbors': 5, 'reg__knr__weights': 'distance', 'reg__rf__n_estimators': 250}
    Best cross-validated R^2 score: 0.9072



```python
gridsearch2.best_estimator_.named_steps['reg'].final_estimator_.coef_
```




    array([0.28070344, 0.15668536, 0.2562573 , 0.34794024])




```python
from sklearn.model_selection import validation_curve


X_train_scaled = preproc_pipe.fit_transform(X_train)
X_train_dense = X_train_scaled.toarray() if hasattr(X_train_scaled, 'toarray') else X_train_scaled

# Fresh copy of reg_boss2 (do NOT reuse fitted one)
estimators2 = [
    ('ridge', RidgeCV()),
    ('knr', KNeighborsRegressor(metric='euclidean', n_neighbors=25)), 
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('hgbr', HistGradientBoostingRegressor())
]

reg_boss2 = StackingRegressor(
    estimators=estimators2,
    final_estimator=GradientBoostingRegressor(
        n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
        random_state=42
    )
)

param_range = [10, 50, 100, 250, 300]       # n_estimators for RF
train_scores, val_scores = validation_curve(
        reg_boss2, X_train_dense, y_train,
        param_name='rf__n_estimators',
        param_range=param_range,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1, 
        verbose=3,
        error_score='raise')

```



```python
# Step 1: Convert negative MSE back to positive MSE
train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

# Optional: also plot standard deviation (error bars)
train_scores_std = np.std(train_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Step 2: Plot
plt.figure(figsize=(8,6))
plt.plot(param_range, train_scores_mean, marker='o', label='Training MSE')
plt.plot(param_range, val_scores_mean, marker='o', label='Validation MSE')

# Error bars (optional, but nice)
plt.fill_between(param_range,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.2)
plt.fill_between(param_range,
                 val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std,
                 alpha=0.2)

plt.xlabel('Random Forest n_estimators')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve: RF Trees inside StackingRegressor')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_19_0.png)
    


This shows that higher doesn't always mean better. Going with 250 here. 


```python
param_range_lr = [0.05, 0.1, 0.15, 0.2]       # n_estimators for RF
train_scores_lr, val_scores_lr = validation_curve(
        reg_boss2, X_train_dense, y_train,
        param_name='hgbr__learning_rate',
        param_range = param_range_lr,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1, 
        verbose=3,
        error_score='raise')

# Step 1: Convert negative MSE back to positive MSE
train_scores_lr_mean = -np.mean(train_scores_lr, axis=1)
val_scores_lr_mean = -np.mean(val_scores_lr, axis=1)

# Optional: also plot standard deviation (error bars)
train_scores_lr_std = np.std(train_scores_lr, axis=1)
val_scores_lr_std = np.std(val_scores_lr, axis=1)

# Step 2: Plot
plt.figure(figsize=(8,6))
plt.plot(param_range_lr, train_scores_lr_mean, marker='o', label='Training MSE')
plt.plot(param_range_lr, val_scores_lr_mean, marker='o', label='Validation MSE')

# Error bars (optional, but nice)
plt.fill_between(param_range_lr,
                 train_scores_lr_mean - train_scores_lr_std,
                 train_scores_lr_mean + train_scores_lr_std,
                 alpha=0.2)
plt.fill_between(param_range_lr,
                 val_scores_lr_mean - val_scores_lr_std,
                 val_scores_lr_mean + val_scores_lr_std,
                 alpha=0.2)

plt.xlabel('HGBR Learning Rate')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve: HGBR Learning Rate inside StackingRegressor')
plt.legend()
plt.grid(True)
plt.show()
```

    
![png](output_21_7.png)
    


.1 is the best learning rate for lowest validation error


```python
param_range_knr = [2, 5, 10, 15, 20]       # n_estimators for RF
train_scores_knr, val_scores_knr = validation_curve(
        reg_boss2, X_train_dense, y_train,
        param_name='knr__n_neighbors',
        param_range = param_range_knr,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1, 
        verbose=3,
        error_score='raise')

# Step 1: Convert negative MSE back to positive MSE
train_scores_knr_mean = -np.mean(train_scores_knr, axis=1)
val_scores_knr_mean = -np.mean(val_scores_knr, axis=1)

# Optional: also plot standard deviation (error bars)
train_scores_knr_std = np.std(train_scores_knr, axis=1)
val_scores_knr_std = np.std(val_scores_knr, axis=1)

# Step 2: Plot
plt.figure(figsize=(8,6))
plt.plot(param_range_knr, train_scores_knr_mean, marker='o', label='Training MSE')
plt.plot(param_range_knr, val_scores_knr_mean, marker='o', label='Validation MSE')

# Error bars (optional, but nice)
plt.fill_between(param_range_knr,
                 train_scores_knr_mean - train_scores_knr_std,
                 train_scores_knr_mean + train_scores_knr_std,
                 alpha=0.2)
plt.fill_between(param_range_knr,
                 val_scores_knr_mean - val_scores_knr_std,
                 val_scores_knr_mean + val_scores_knr_std,
                 alpha=0.2)

plt.xlabel('KNR Neighbors')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve: KNR Neighbors inside StackingRegressor')
plt.legend()
plt.grid(True)
plt.show()
```
    
![png](output_23_7.png)
    


10 seems to be good. 

## Now, we pick  fav lasso...

## Then, we pick ouourr fav OLS....

## Then, our fav XGBoost

## Then, our fav ?

## Then we pick amongst them or "stack" them into an ensemble

## Then we train the our fav-of-favs or ensemble on all of X_train, y_train, and see how it does on y_test (ok?)

## How does this do on my "holdout" (which is X_test/y_test)


```python
# train the best model on the full training set
best_model = gridsearch.best_estimator_
best_model.fit(X_train, y_train)
# evaluate the model on the test set
y_pred = best_model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
r2
```




    0.9021556688417908




```python
# train the best model on the full training set
best_model2 = gridsearch2.best_estimator_
best_model2.fit(X_train, y_train)
# evaluate the model on the test set
y_pred2 = best_model2.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
r2_1 = r2_score(y_test, y_pred2)
r2_1
```




    0.9003065015494791



Go with the better model. 


```python
best_model = max([gridsearch.best_estimator_, gridsearch2.best_estimator_], key=lambda x: x.score(X_test, y_test))
```

## Now train the model in full and make predictions on new data


```python
# now estimate that on the full data, to be applied on the real holdout data (input_data2/housing_holdout.csv)
# fit not on X_train, y_train but on the full data
best_model.fit(housing, y)

```


Fit the model from the solutions file from last assignment on all the data
Load and .predict using the holdout data and store those predictions as a variable.
Your answer file for assignment 8 should have 990 rows of (989 of data + 1 header row).
You can use pandas functions you know and love to output the parcel ID variable and your predictions.


```python

# create predictions 
holdout = pd.read_csv('input_data2/housing_holdout.csv') # load the new holdout data

holdout_X_vals =  holdout.drop('parcel', axis=1) # drop the parcel number, not a feature

y_pred = best_model.predict(holdout_X_vals) # make predictions!

# save for output: parcel number + y_pred to csv 
df_out = pd.DataFrame({'parcel':holdout['parcel'],
                       'prediction':y_pred})

df_out.to_csv('submission/MY_PREDICTIONS.csv',index=False)

# open it... does it look like the sample version? 
```
