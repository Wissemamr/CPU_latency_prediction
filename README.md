# CPU Latency Prediction Using a Regression Model with Ensemble Methods 


## Definition
Ensemble methods combine predictions from multiple models to improve overall prediction accuracy and robustness.

## Approach

- In our case, we used ensemble methods in order to enhance the coefficient of determination $R^2$ and mitigate individual model biases and variances for the predictions of **CPU latency** in Cloud environments.
- Initially, 4 algorithms including **LGBMRegressor**, **XGBRegressor**, **RandomForestRegressor**, and **CatBoostRegressor** were trained on the dataset. Each model captures different aspects of the data.
- Post-training, predictions from the **4** models were averaged using a simple averaging approach


## Evaluation metric
The coefficient of determination $R^2$ was used to assess the models' performances. 

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$


$\sum_{i}(y_i - \hat{y}_i)^2 \quad \text{represents the Residual Sum of Squares (RSS).}$

$
\sum_{i}(y_i - \bar{y})^2 \quad \text{represents the Total Sum of Squares (TSS).}
$
## Results

The results are summarized in the following table


| Model            | Training R²     | Validation R²     |
|------------------|-----------------|-------------------|
| `LGBM`           | **0.9830**      | **0.9741**        |
| `XGBoost`        | **0.9961**      | **0.9723**        |
| `RandomForest`   | **0.9956**      | **0.9747**        |
| `CatBoost`       | **0.9906**      | **0.9756**        |





| Dataset       | Average R²     |
|---------------|----------------|
| Training      | **0.9913**     |
| Validation    | **0.9742**     |



**Note** : Big thanks to **SoAI** for suggesting this fun challenge !