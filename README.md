LogisticRegression() : 
Training Accuracy :  0.5108950915919117
Validation Accuracy :  0.4793388429752066

SVC(kernel='poly', probability=True) : 
Training Accuracy :  0.49871262935205074
Validation Accuracy :  0.5042644628099173


XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...) :
Training Accuracy :  0.9428046640536596
Validation Accuracy :  0.5222809917355372


![EDA fig 1](https://github.com/user-attachments/assets/542254cd-e5d9-4f17-b3c5-1711ef64efa3)
- Exploratory data analysis for the NIFTY 500 RELIANCE stocks over the time period and we can see that the graph gradually increases.

![EDA fig 2](https://github.com/user-attachments/assets/dd06ea58-cdd0-47bb-a943-54b314a4fa8a)
- Two peaks in the OHLC data distribution plot indicate that there have been large variations in the data in two different regions. The volume data is also skewed to the left.

 ![EDA fig 3](https://github.com/user-attachments/assets/8c3c657e-d715-46e2-903a-bd6f7690dd48)
- While there are outliers in the volume data, there are none in the remaining columns.

![EDA fig 4 bar graph](https://github.com/user-attachments/assets/be5e33c5-56a2-43e9-95a3-5dfdddfabd98)
- Bar graph indicates the rise in the stock price over the time duration of the given years

![Pie chart](https://github.com/user-attachments/assets/cca6ebb5-436e-4b0e-afc1-d07cbd9bb863)
- When adding features to our dataset, we must make sure that there aren't any strongly linked characteristics because they hinder the algorithm's learning process.

![Heatmap](https://github.com/user-attachments/assets/55da20f1-597a-4a5a-b802-dc556efb0ce3)
- We can conclude from the heatmap above that there is a strong correlation between OHLC, which is fairly evident, and that the newly added features do not have a strong correlation with one another or with the previously supplied data, indicating that we can proceed with building our model.

![Probability analysis](https://github.com/user-attachments/assets/4583c621-2296-498e-b7f6-04cec4a060e2)
- Probability analysis of the stocks price increase at the quarter end is a NO where as at non quarter end it increases with a less significant difference, bit strange but the data fetched is from the RELIANCE.

![Moving avg](https://github.com/user-attachments/assets/147c4a4f-88dd-48bd-a701-d8eb40c30a3a)
- Lagged Feature, technical indicator RSI & MACD

![Volume change](https://github.com/user-attachments/assets/7f5a7ce0-d1db-4bd6-a102-276acf4e243f)
- Volume changes of the absolute values of the data fetched after the EDA analysis.

