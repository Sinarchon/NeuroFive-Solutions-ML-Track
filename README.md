# Neurofive Solutions — Machine Learning Track Portfolio

This repository contains my complete work and deliverables for the Neurofive Solutions Machine Learning Internship Track, structured week-by-week from foundational exploratory data analysis to production-grade deep learning web deployment.

---

## Live Capstone Deployment
* **Deep Emotion & Sarcasm-Aware Sentiment Analyzer:** [https://neurofive-solutions-ml-track-capstone.streamlit.app/](https://neurofive-solutions-ml-track-capstone.streamlit.app/)

---

## Weekly Breakdown & Task Index

### Week 1 & 2: Exploratory Data Analysis & Data Cleaning
* **Dataset:** Titanic Survival Dataset & California Housing.
* **EDA & Inspection:** Inspected dataframe structures using `.info()`, `.describe()`, and `.head()` to identify missing values and data types.
* **Data Cleaning Strategies:** 
  * Handled missing numerical values (e.g., `Age`) using the median for outlier resilience.
  * Filled categorical missing values (e.g., `Embarked`) using the mode.
  * Dropped heavily sparse columns (e.g., `Cabin`) to prevent noise injection.
* **Data Visualization:** Built histograms, outlier boxplots, and correlation heatmaps using `matplotlib` and `seaborn`.

---

### Week 3: Baseline Classification & Regression Modeling
* **Classification (Titanic):** Encoded categorical features via `pd.get_dummies()`, split data using `train_test_split`, and trained an initial baseline **Logistic Regression** classifier.
* **Model Evaluation:** Analyzed baseline performance using accuracy scores and confusion matrices.
* **Regression (Housing):** Selected core predictive features (`MedInc`, `AveRooms`, `HouseAge`, `AveOccup`) and trained a baseline **Linear Regression** model, evaluating performance via RMSE and R² score.

---

### Week 4: Feature Engineering, Pipelines & Ensemble Learning
* **Advanced Feature Engineering:** Engineered behavioral features such as `FamilySize` (combining `SibSp`, `Parch`, and the passenger) and `IsAlone` (binary flag).
* **Scikit-Learn Pipelines:** Automated data preprocessing using `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` to completely eliminate data leakage.
* **Model Serialization:** Saved production-ready pipelines using `joblib` (`titanic_pipeline.pkl`).
* **Ensemble Learning (Random Forest vs. XGBoost):**
  * Trained and benchmarked advanced ensemble methods against baseline models.
  * **Results Summary:** Random Forest achieved top accuracy (~83.24%), outperforming XGBoost (~81.01%) and Logistic Regression (~79.89%).
  * **Feature Importances:** Visualized predictive weights using horizontal feature importance plots.

---

### Week 5: Handling Imbalanced Data & Production Web Deployment
* **Handling Imbalanced Real-World Data (Telco Churn):**
  * Analyzed class distribution (~73% stayed vs. ~27% churned) using distribution bar charts.
  * Proved why **Accuracy is a misleading metric** for skewed datasets (a lazy model predicting "No Churn" hits 73% accuracy without catching any churners).
  * Implemented `class_weight='balanced'` to shift model focus and heavily optimize minority-class **Recall** and **F1-score**.
* **Model Deployment (Streamlit Web Apps):**
  * Shipped an interactive Titanic Survival prediction app using the serialized joblib pipeline.
  * **Capstone Project:** Built and deployed a **Deep Emotion & Sarcasm-Aware Sentiment Analyzer** utilizing a fine-tuned Hugging Face transformer model (`DistilRoBERTa`) combined with custom heuristic validation to decode multi-class emotional nuance and cynical tone.

---

## Tech Stack
* **Languages:** Python 3.13
* **Machine Learning & NLP:** Scikit-Learn, XGBoost, PyTorch, Hugging Face Transformers
* **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn
* **Deployment & Serialization:** Streamlit Community Cloud, Joblib

---

## Capstone Project

> **Deep Emotion & Sarcasm-Aware Sentiment Analyzer:** This production-grade natural language processing application goes beyond traditional binary sentiment classification by integrating a fine-tuned Hugging Face transformer model (`DistilRoBERTa`) to decode complex emotional nuance. Capable of multi-class emotion detection across joy, anger, sadness, disgust, fear, surprise, and neutral states, the app also features custom heuristic rules designed to identify sarcasm—successfully flagging instances where superficially positive vocabulary masks heavy cynical or negative underlying sentiment.

* `Task#09.ipynb` — Telco churn class imbalance analysis and class weighting implementation.
* `capstone.py` — Streamlit source code for the Deep Emotion & Sarcasm-Aware Sentiment Analyzer.
* `requirements.txt` — Project environment and dependencies.
