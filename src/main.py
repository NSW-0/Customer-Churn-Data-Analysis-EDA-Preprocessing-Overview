import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
# ---Load data---
df = pd.read_csv("Customer_data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
# ---Fill missing values---
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
df["SupportCalls"] = df["SupportCalls"].fillna(df["SupportCalls"].mode()[0])
# ---Set plot style---
sns.set_style("whitegrid")
numeric_cols = df.select_dtypes(include='number').columns
# --- Boxplots of all numeric columns before outlier removal ---
n_cols = 2
n_rows = math.ceil(len(numeric_cols) / n_cols)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], color="skyblue", ax=ax)
    ax.set_title(f"Boxplot of {col}")
    ax.set_xlabel("")
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()
# --- fix outliers for income using IQR ---
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1
Upper_Limit = Q3 + 1.5 * IQR
Lower_Limit = Q1 - 1.5 * IQR
Income_mean=df['Income'].mean()
df['Income'] = df['Income'].apply(lambda x:Income_mean if x < Lower_Limit or x > Upper_Limit else x)
# --- fix outliers for income using IQR ---
Q1_Age = df['Age'].quantile(0.25)
Q3_Age = df['Age'].quantile(0.75)
IQR_Age = Q3_Age - Q1_Age
Upper_Limit_Age = Q3_Age + 1.5 * IQR_Age
Lower_Limit_Age = Q1_Age - 1.5 * IQR_Age
df = df[(df['Age'] >= Lower_Limit_Age) & (df['Age'] <= Upper_Limit_Age)]
# --- fix outliers for SupportCalls using IQR ---
Q1_SupportCalls = df['SupportCalls'].quantile(0.25)
Q3_SupportCalls = df['SupportCalls'].quantile(0.75)
IQR_SupportCalls = Q3_SupportCalls - Q1_SupportCalls
Upper_Limit_SupportCalls = Q3_SupportCalls + 1.5 * IQR_SupportCalls
Lower_Limit_SupportCalls = Q1_SupportCalls - 1.5 * IQR_SupportCalls
SupportCalls_median=df['SupportCalls'].median()
df['SupportCalls'] = df['SupportCalls'].apply(lambda x:SupportCalls_median if x < Lower_Limit_SupportCalls or x > Upper_Limit_SupportCalls else x)
# --- Boxplots after outlier removal ---
n_cols_per_row = 2
n_rows = math.ceil(len(numeric_cols) / n_cols_per_row)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols_per_row, figsize=(12, 4*n_rows))
axes = axes.flatten()
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], color="skyblue", ax=ax)
    ax.set_title(f"Boxplot of {col} (After Outlier Handling)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(col)
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()
# --- Scale numeric features ---
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df.head())
# --- Histograms of all numeric columns ---
fig, axes = plt.subplots(nrows=len(numeric_cols), figsize=(6, 4*len(numeric_cols)))
if len(numeric_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col], bins=10, color='cyan', ax=ax)
    ax.set_title(f"Histogram of {col}")
plt.tight_layout()
plt.show()
# ---Analyze the distribution of categorical variables using bar plots---
cat_cols = ['Gender', 'ProductType']
for col in cat_cols:
    plt.figure(figsize=(5,4))
    sns.barplot(data=df, x=col, y='ChurnStatus', estimator='mean', errorbar=None)
    plt.title(f"Average Churn Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Average Churn Rate")
    plt.xticks(rotation=45)
    plt.show()
# --- Boxplots vs ChurnStatus ---
numeric_cols_no_target = [col for col in numeric_cols if col != 'ChurnStatus']
fig, axes = plt.subplots(nrows=len(numeric_cols_no_target), figsize=(6, 4*len(numeric_cols_no_target)))
if len(numeric_cols_no_target) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_cols_no_target):
    sns.boxplot(x='ChurnStatus', y=col, data=df, color='cyan', ax=ax)
    ax.set_title(f"{col} vs ChurnStatus")
plt.tight_layout()
plt.show()
# ---Investigate relationships between categorical variables and the target using bar plots---
for col in cat_cols:
    plt.figure(figsize=(4, 4))
    sns.barplot(data=df, x=col, hue='ChurnStatus', estimator='mean', errorbar=None)
    plt.title(f"{col} vs ChurnStatus")
    plt.xticks(rotation=45)
    plt.show()
# --- Correlation heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()
# --- Pairplot ---
sns.pairplot(df, vars=numeric_cols_no_target, hue='ChurnStatus', diag_kind='kde')
plt.suptitle("Pairwise Relationships between Numerical Features", y=1.02)
plt.show()