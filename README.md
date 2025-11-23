# Housing price prediction

As part of my data science bootcamp, I completed a project for a consulting firm in the real estate sector. The goal was to use machine learning to determine the true value of real estate more accurately and efficiently.

The project was based on an extensive data set of real home sales in Ames, Iowa, including around 80 features per property (e.g. area, condition, amenities).

### Two key tasks:

- Classification - predicting whether a home is “expensive” or “not expensive” (categorical target).
- Regression - prediction of the exact sales price (variable `SalePrice`) in dollars (numerical target).


## Classification


The classification part of the project is structured in a pipeline approach, starting with data preprocessing and ending with the model evaluation.


#### 1. Data Preprocessing
First, a feature selection step is applied to reduce noise and improve model interpretability. Missing values are handled using a SimpleImputer.
Next, the data is divided into numerical and categorical features since they are processed in different sub-pipelines. For encoding categorical data, two types of encoders are used:

- ordinal encoder: for ordered features like condition or quality (e.g. `Ex`, `Gd`, `TA`, ... → mapped to 5 → 4 → 3, etc.)
- One-Hot Encoding for nominal, unordered features (e.g. neighborhood, building type)

All preprocessing steps are combined using a ColumnTransformer.

#### 2. Random Forest Classifier
The preprocessed data is then passed to a Random Forest Classifier which works by building multiple decision trees and aggregating their predictions.

#### 3. Hyperparameter tuning
To optimize model performance, a Randomized Search Cross-Validation (`RandomizedSearchCV`) is used. 
Tuned parameter:
- `n_estimators` (number of trees in the forest)
- `max_depth` (maximum depth of each tree)
- `min_samples_split` (minimum number of samples required to split an internal node)
- `min_samples_leaf` (minimum number of samples required to be at a leaf node)
- `ccp_alpha` (cost complexity pruning parameter)

#### 4. Evaluation
After training the model on the training data with the best parameters, the model is used to predict the housing prices for the unseen test set, with the accuracy of this prediction expressed through the accuracy score.


<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/acejolanda/supervised_housing/blob/main/pipeline.png?raw=true">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/acejolanda/supervised_housing/blob/main/pipeline.png?raw=true">
    <img alt="Schema Diagramm" src="https://github.com/acejolanda/supervised_housing/blob/main/pipeline.png?raw=true" width="70%">
  </picture>
</div>

## Regression

To predict the exact sales prices of the houses data set, a modular pipeline approach is chosen as well:

#### 1. Feature Engineering and Data Preprocessing
Data preprocessing including feature selection, imputation, and encoding - is handled analogously to the classification task, using a ColumnTransformer setup. For the regression pipeline, additional features are created and added on top of those used in the classification part, e.g., “TotalSF” which is the sum of all individual surfaces.
There is also an option to check feature skewness, although this functionality is currently included for analysis purposes only.  
The target variable `SalePrice` is log-transformed using `np.log1p` to reduce skewness and stabilize the distribution.

All preprocessing steps are combined using a ColumnTransformer.

#### 2. Regression model selection
The next step is selecting the regression model for the pipeline. Three models are evaluated each in a complete pipeline with preprocessing:

- k-Nearest Neighbors Regressor (KNN)
- Random Forest Regressor
- XGBoost Regressor

The model is selected when instantiating the pipeline using the `build_model()` method.

#### 3. Training and Hyperparameter-Tuning
The model is then trained, and hyperparameters are tuned using either RandomizedSearchCV or GridSearchCV.
The search space varies depending on the model:


- KNN: `n_neighbors`, `weights`, distance metric (`p`), number of selected features (`k`)
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- XGBoost: `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` and other

Hyperparameter optimization is based on the R² score using cross-validation.

#### 4. Evaluation
The predictions are back-transformed (`expm1`) and evaluated using the R² score on the test set. Scatterplots are also created showing true vs. predicted prices, including a diagonal reference line as the ideal line.
This approach is used for the model with knn (with grid search for hyperparameter tuning), for random forest (with random search), and for XGBoost (with grid search). XGBoost delivers the best results with an R2 score of approximately 0.9.