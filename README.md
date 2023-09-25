# FRAUD-DETECTION-MACHINE-LEARNING

1.	**Data cleaning including missing values, outliers, and multi-collinearity.**
   
Ans Data Cleaning - Replacing Categories with Numbers:
    In this step, I have replaced transaction types ('PAYMENT', 'TRANSFER', etc.) with numerical codes (2, 4, 1, 5, 3). This helps the model work with categorical data more effectively.
     Data Cleaning - Dropping Unnecessary Columns
  I have removed certain columns ('step', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud') from the dataset. These columns are not needed for our analysis and can be dropped to simplify the data and reduce noise.

3. **Describe your fraud detection model in elaboration**
I have done many thing with this dataset,Firstly I have cleaned the data as there is no missing and duplicate value .
I have dropped some column and replaced ‘type’ column with a numerical number for better accuracy.

EDA- Exploratory Data Analysis 
I have done EDA on type and isFraud column , and found that transfer transaction is mostly use for farud .

**Data Splitting** 
As in this dataset the target variable  is ‘isFraud’ as it will help us to find either farud is happening and
how much is happening

**Train model **
I have trained my model using RandomForest Classifier 

**Model Evaluation **

I have used classification report to mae a report on accuracy,precision 

**
3.How did you select variables to be included in the model?**

 The 'isFraud' column is the dependent variable in a fraud detection model, representing whether a transaction 
 is fraudulent (1) or not (0). It serves as the target variable that the model aims to predict based on other features 
 and patterns in the dataset,making it the variable of interest in the analysis.
 **
 
4. Demonstrate the performance of the model by using best set of tools.**
Classification report:               precision    recall  f1-score   support

           0       1.00      1.00      1.00   1270945
           1       0.88      0.91      0.89      1579

    accuracy                           1.00   1272524
   macro avg       0.94      0.96      0.95   1272524
weighted avg       1.00      1.00      1.00   1272524
**
5. What are the key factors that predict fraudulent customers?**

 Unusual transaction behavior (frequency and amounts).
Geographic anomalies in transactions.
Suspicious account activity (creation, login patterns, balances).
Specific transaction types (e.g., 'TRANSFER' or 'CASH_OUT'). Deviations from historical behavior.
 Account-to-account interactions (unusual transfers or withdrawals).

6.**Do these factors make sense? If yes, How? If not, How not?**

Yes, these factors make sense for fraud prediction because they capture common characteristics and behaviors associated
with fraudulent activities, helping the model identify suspicious transactions effectively.
**
7.What kind of prevention should be adopted while company update its infrastructure?**

Prevention against fraud during infrastructure updates is important for any organization.
By following some of the steps like security measures, staying vigilant, and educating both employees and users,
a company can reduce the risk of fraudulent activities.
It's an ongoing process that requires adapting to evolving threats and maintaining a strong culture of security awarenes.

**
8. Assuming these actions have been implemented, how would you determine if they work?**
We need to check on a continuous and collect data and again implement machine learning to know
 how much it reduces and how much more need to work ."""
 
