
# Assessing Loan Safety: A Supervised Machine Learning Approach


## Overview of the Analysis

The purpose of this analysis was to utilize supervised machine learning to predict whether loans would be classified as risky (default) or safe (fully paid), based on various financial data. This information assists lenders in assessing the risk associated with issuing loans and making informed decisions regarding loan approvals.

The dataset comprised lending information, including features such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and loan status. We focused on loan status as the dependent variable, with all other features acting as independent factors that influence this status.

## Analysis

We began by examining the variables of interest using `value_counts` to identify which values corresponded to `0` (healthy loan) and `1` (high-risk loan). Knowing the real-life outcomes based on these values allowed us to implement a model that accurately predicts future data points.

The chosen model for this analysis was **Logistic Regression**, a classification algorithm used to predict the probability of a binary outcome (two classes), such as whether a loan will default or not. In our case, the two classes are represented by `0` (healthy loan) and `1` (high-risk loan).

We split our independent variables (`X`) and dependent variable (`y`) into training and testing datasets, assigning a random state of 1 to ensure reproducibility. We then fit the model to normalize the results. After fitting, we ran the `predict` function on our independent variables, saving the training predictions into the testing data labels. Following this, we created a confusion matrix using `y_test` (the dependent variable) and `testing_predictions` (the predicted labels). The resulting matrix values were:

```
[[18655   110]  // TN      FP
 [   36   583]] // FN      TP
```

Finally, we generated a classification report for the logistic regression model to evaluate its utility in predicting loan defaults.

```
                precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.94      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```

## Results

The confusion matrix indicates a high number of correctly predicted safe loans, with some misclassifications for both safe and risky loans. The model performs well in predicting risky loans, as evidenced by the precision and recall metrics.

- The logistic regression model predicts healthy loans with **100% precision**.
- The precision for class `1` (high-risk loans) is **0.84**, meaning that 84% of instances predicted as `1` were indeed `1`.
- The recall for healthy loans is **99%**, while for high-risk loans, it is **94%**.
- The F1 score, which balances precision and recall, indicates that this model is effective for predicting outcomes, particularly for healthy loans, with a score of **100%**.

## Summary

The logistic regression model demonstrates excellent performance, especially in predicting binary classes. It would be interesting to explore how this model performs on data that does not adhere to the `0` and `1` classifications.

### End Notes

This assignment was completed with references to past class activities and the assistance of ChatGPT.
