# AI-ML Assignment 2: Customer Churn Prediction using Logistic Regression

## Objective
The objective of this assignment is to develop and evaluate a **Logistic Regression** model using Pandas and Scikit-Learn to predict customer churn for a telecommunications company based on demographic information and service usage patterns.

---

## Dataset Link
- **Dataset**: Telco Customer Churn Dataset
- **Source**: [Kaggle Telco Customer Churn Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- *Note: As per submission instructions, the raw dataset file is not uploaded directly to this repository. Please download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the link above and place it in the project root folder before running the code.*

---

## Libraries Used
- `pandas` - Data loading, cleaning, and preprocessing
- `numpy` - Array manipulation and numerical operations
- `scikit-learn` - Model building (`LogisticRegression`), data splitting (`train_test_split`), scaling (`StandardScaler`), and metrics evaluation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`)
- `matplotlib` & `seaborn` - Evaluation visualizations

---

##  Methodology

1. **Task 1: Data Understanding**
   - Dataset loaded using Pandas.
   - Initial 5 records displayed.
   - Identified **Target Variable**: `Churn`
   - Identified **Numerical Features**: `tenure`, `MonthlyCharges`, `TotalCharges`
   - Identified **Categorical Features**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

2. **Task 2: Data Preprocessing**
   - Checked for missing values (`TotalCharges` contained whitespace strings for new customers with zero tenure).
   - Handled missing values by converting `TotalCharges` to numeric and imputing missing values with the median.
   - Encoded the target variable `Churn` (`Yes` -> 1, `No` -> 0).
   - Applied One-Hot Encoding (`pd.get_dummies`) to convert categorical variables into numerical format.
   - Split dataset into **80% training set** and **20% testing set** (`random_state=42`, stratified).
   - Standardized feature distributions using `StandardScaler`.

3. **Task 3: Model Development**
   - Built a `LogisticRegression` classification model (`max_iter=1000`, `random_state=42`).
   - Trained the model on the 80% training subset.
   - Made customer churn predictions on the 20% test subset.

4. **Task 4: Model Evaluation**
   - Measured model metrics on the test dataset:
     - **Accuracy Score**: ~80.5%
     - **Precision Score**: ~66.0%
     - **Recall Score**: ~54.0%
     - **F1-Score**: ~59.3%
   - Generated and analyzed the **Confusion Matrix**.

---

## Key Observations & Results

1. **Overall Performance**: The model achieves an overall accuracy of **80.5%**, providing a solid baseline for customer churn classification.
2. **Precision vs. Recall**: Precision (66.0%) is noticeably higher than Recall (54.0%), indicating that while the model is accurate when it predicts a customer will churn, it fails to identify approximately 46% of all actual churners.
3. **Impact of Class Imbalance**: Churners represent roughly 26.5% of total dataset records, which shifts predictions slightly toward the majority class (No Churn).

---

## Conclusion

In this assignment, a Logistic Regression model was developed to predict telecommunications customer churn using demographic and service subscription data. Key findings reveal that customer retention is strongly influenced by contract terms and tenure duration; customers on short month-to-month contracts and electronic check payments show significantly higher churn rates than long-term account holders.

The trained model achieved an overall accuracy of 80.5% with a precision of 66%. However, a key limitation of Logistic Regression in this scenario is its reliance on a linear decision boundary, which prevents it from automatically capturing complex non-linear feature interactions without manual feature transformation. Furthermore, class imbalance within the dataset reduces the model's recall for churned customers. Future modeling efforts could leverage non-linear algorithms such as Random Forests or Gradient Boosting alongside resampling methods like SMOTE.
