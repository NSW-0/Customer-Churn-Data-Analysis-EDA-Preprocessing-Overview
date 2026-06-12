# Customer Churn Data Analysis (EDA & Preprocessing)

##  Overview

This project focuses on **data preprocessing and exploratory data analysis (EDA)** on a customer dataset to understand patterns related to **customer churn**.

The workflow includes handling missing values, detecting and treating outliers, scaling features, and visualizing relationships between variables to extract meaningful insights.

---

##  Objectives

* Clean and preprocess raw data
* Handle missing values and outliers
* Normalize numerical features
* Perform univariate and bivariate analysis
* Identify patterns related to customer churn

---

##  Dataset Description

The dataset simulates customer information including:

* **CustomerID** – Unique identifier
* **Age** – Customer age
* **Gender** – Male / Female
* **Income** – Annual income
* **Tenure** – Years with the company
* **ProductType** – Basic / Premium
* **SupportCalls** – Number of support calls
* **ChurnStatus** – Target variable (0 = stayed, 1 = churned)

---

##  Preprocessing Steps

### 1. Data Cleaning

* Checked missing values using `.isnull()`
* Filled missing values:

  * Median → Age, Income, Tenure
  * Mode → SupportCalls

### 2. Outlier Handling

* Used **IQR (Interquartile Range)** method
* Income → replaced outliers with mean
* Age → removed outliers
* SupportCalls → replaced outliers with median

### 3. Feature Scaling

* Applied **Min-Max Scaling** to normalize numerical features

---

##  Exploratory Data Analysis (EDA)

###  Univariate Analysis

* Histograms for numerical features
* Boxplots before and after outlier handling

###  Bivariate Analysis

* Boxplots vs **ChurnStatus**
* Bar plots for:

  * Gender vs Churn
  * ProductType vs Churn

###  Correlation Analysis

* Heatmap showing relationships between numerical features

###  Pairwise Relationships

* Pairplot to visualize feature interactions

---

##  Visualizations Included

* Boxplots (before & after outliers)
* Histograms
* Bar plots (categorical vs churn)
* Correlation heatmap
* Pairplot

---

##  How to Run

### 1. Install dependencies

```bash
pip install pandas seaborn matplotlib scikit-learn
```

### 2. Run the script

```bash
python main.py
```

---

##  Project Structure

```
.
├── main.py
├── Customer_data.csv
├── Report.pdf
└── README.md
```

---

##  Concepts Used

* Data Cleaning
* Missing Value Imputation
* Outlier Detection (IQR)
* Feature Scaling (Min-Max)
* Data Visualization (Seaborn, Matplotlib)

---

##  Key Insights (Example)

* Customers with higher support calls tend to churn more
* Income and tenure show patterns related to customer retention
* Certain product types may have higher churn rates

---

##  References

###  Project Files

* [ Source Code (main.py)](src/main.py)
* [ Dataset (customer_data.csv)](data/customer_data.csv)
* [ Report (Report.pdf)](doc/Report.pdf)

---

###  Official Documentation

* [Pandas](https://pandas.pydata.org/docs/)
* [Matplotlib](https://matplotlib.org/stable/contents.html)
* [Seaborn](https://seaborn.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

---

##  Notes

This project was developed as part of a **Machine Learning and Data Science assignment**, focusing on preprocessing and exploratory data analysis techniques.
