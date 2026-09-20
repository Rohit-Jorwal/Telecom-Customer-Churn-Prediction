# Telecom Customer Churn Prediction
Machine learning project for analyzing and predicting telecom customer churn.  
This is a machine learning project to analyze customer behavior, identify factors associated with telecom customer churn, and compare classification models for predicting customer churn.

## 📌 Business Problem

Customer churn is an important challenge for telecom companies because retaining existing customers can be more cost-effective than acquiring new ones.

The objective of this project is to:

* Analyze customer behavior and characteristics
* Explore factors associated with customer churn
* Prepare customer data for machine learning
* Build and compare different churn prediction models
* Evaluate model performance using multiple classification metrics

## 📊 Dataset

The project uses two datasets:

* `Client.csv`
* `Record.csv`

The datasets are merged using `Customer_ID`.

```python
df = client.merge(record, on="Customer_ID")
```

The target variable is:

* `churn` — indicates whether a customer churned.

## 🔍 Exploratory Data Analysis

The project explores customer churn through:

* Dataset structure and summary statistics
* Missing-value analysis
* Churn distribution
* Customer tenure
* Total revenue
* Average monthly usage
* Total minutes used
* Revenue distribution
* Correlation analysis

### Visualizations

The notebook includes:

* Missing-value percentage bar chart
* Churn distribution plot
* Customer tenure vs. churn boxplot
* Total revenue vs. churn boxplot
* Average monthly minutes vs. churn boxplot
* Total minutes used boxplot
* Revenue distribution histogram
* Correlation heatmap
* Confusion matrices for the classification models

## 🧹 Data Cleaning

Missing values are handled separately for numerical and categorical variables.

### Numerical variables

Missing numerical values are replaced using the median.

### Categorical variables

Missing categorical values are replaced using the mode.

## ⚙️ Feature Preparation

The target variable `churn` is separated from the input features.

Categorical and numerical features are handled separately.

The preprocessing pipeline uses:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features

## 🤖 Machine Learning Models

Three classification models are developed and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Logistic Regression

A Logistic Regression model is implemented using a preprocessing pipeline.

### Decision Tree

A Decision Tree classifier is implemented with a maximum depth of 8.

### Random Forest

A Random Forest classifier is implemented with:

* 300 trees
* `random_state=42`
* Parallel processing using `n_jobs=-1`

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The results are combined into a comparison table and sorted according to ROC-AUC.

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 📁 Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── Telecom_Customer_Churn.ipynb
├── Telecom_Customer_Churn.py
├── Telecom_Customer_Churn_Report.pdf
└── README.md
```

## 🚀 How to Run

1. Clone or download this repository.
2. Place the required `Client.csv` and `Record.csv` datasets in the working directory.
3. Install the required Python libraries.
4. Open `Telecom_Customer_Churn.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
5. Run the notebook cells sequentially.

## 🎯 Objective

The objective of this project is to use exploratory data analysis and machine learning techniques to study telecom customer churn and compare different classification models for churn prediction.
