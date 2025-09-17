# Housing price prediction

As part of my data science bootcamp, I completed a project for a consulting firm in the real estate sector. The goal was to use machine learning to determine the true value of real estate more accurately and efficiently.

The project was based on an extensive data set of real home sales in Ames, Iowa, including around 80 features per property (e.g. area, condition, amenities).

### Two key tasks:

- Classification - predicting whether a home is “expensive” or “not expensive” (categorical target).
- Regression - prediction of the exact sales price in dollars (numerical target).


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
