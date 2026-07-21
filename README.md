# Customer Churn Prediction using Logistic Regression

## Objective
Develop and evaluate a **Logistic Regression** model using Pandas and Scikit-Learn to predict customer churn for a telecommunications company based on demographic details and service usage patterns.

---

## Dataset Link
- **Dataset**: Telco Customer Churn Dataset
- **Source**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Libraries Used
- `pandas` - Data manipulation and preprocessing
- `numpy` - Numerical computing
- `scikit-learn` - Data splitting, scaling, Logistic Regression modeling, and evaluation metrics
- `matplotlib` & `seaborn` - Visualization and confusion matrix heatmap

---

## Methodology

1. **Task 1: Data Understanding**
   - Loaded dataset and inspected initial records.
   - Identified numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`), categorical features (`gender`, `Contract`, `InternetService`, etc.), and target variable (`Churn`).

2. **Task 2: Data Preprocessing**
   - Converted `TotalCharges` to numeric float values and imputed missing entries using median imputation.
   - Encoded binary target variable (`Yes` -> 1, `No` -> 0).
   - Applied One-Hot Encoding (`pd.get_dummies`) on categorical features.
   - Split dataset into 80% training and 20% testing sets using stratification.
   - Standardized features with `StandardScaler`.

3. **Task 3: Model Development**
   - Trained a `LogisticRegression` model on the 80% training subset.
   - Generated predictions on the 20% test subset.

4. **Task 4: Model Evaluation**
   - Measured test performance metrics:
     - **Accuracy**: ~80.5%
     - **Precision**: ~66.0%
     - **Recall**: ~54.0%
     - **F1-Score**: ~59.3%
   - Evaluated the confusion matrix.

---

## Observations & Results

1. **Overall Accuracy**: The model achieves an accuracy of ~80.5% on test data.
2. **Precision vs. Recall**: Precision (66.0%) is higher than recall (54.0%), showing good accuracy for positive churn predictions while missing some actual churners due to class imbalance.
3. **Primary Churn Factors**: Short contract length (month-to-month) and lower tenure are key predictors of customer churn.

---

## Conclusion

In this project, a Logistic Regression model was built to predict customer churn using telecom subscription data. The analysis shows that customer churn is strongly linked to contract duration and tenure length, with month-to-month contract holders exhibiting the highest risk of churn.

The model achieved an accuracy of 80.5% and a precision of 66%. A main limitation of Logistic Regression here is its assumption of a linear decision boundary, which prevents it from naturally modeling complex feature interactions without manual feature engineering. Class imbalance also lowers recall for churned customers. Future work could test tree-based models like Random Forests along with sampling techniques like SMOTE.
