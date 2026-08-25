# Week 2: Titanic Survival Classification

**Approach:** 
Cleaned the dataset by filling missing ages with the median and dropping heavily missing columns like Cabin. Converted categorical variables (Sex, Embarked) into numeric format using `pd.get_dummies()`. Finally, trained a Logistic Regression model on an 80/20 train-test split.

**Final Accuracy:** ~81%