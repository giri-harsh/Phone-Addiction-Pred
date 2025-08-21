#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#%%


pd.set_option("display.max_columns", None)   
pd.set_option("display.width", None)        

# %%
df = pd.read_csv("teen_phone_addiction_dataset.csv")
# %%
df.info()
# %%
# to drop -> id, name 
# map -> gender , location , school grade , phone use purpose

#%%

df = df.drop(columns=["ID", "Name","Location"],axis=1)
# %%
df.head()
# %%
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cols = ['Gender', 'School_Grade', 'Phone_Usage_Purpose']
ohe = OneHotEncoder(sparse_output=False,drop=None)
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols)) 
df_encoded = pd.concat([df.drop(columns = cols), encoded_df], axis=1)


# %%
df_encoded.head()
# %%


###########################
#iqr for outliers
# %%
# not include in iqr -> Gender_Female	Gender_Male	Gender_Other	School_Grade_10th	School_Grade_11th	School_Grade_12th	School_Grade_7th	School_Grade_8th	School_Grade_9th	Phone_Usage_Purpose_Browsing	Phone_Usage_Purpose_Education	Phone_Usage_Purpose_Gaming	Phone_Usage_Purpose_Other Phone_Usage_Purpose_Social Media"
#%%
# 
# Clean column names first
numeric = df_encoded.drop(
    [
        "Gender_Female", "Gender_Male", "Gender_Other",
        "School_Grade_10th", "School_Grade_11th", "School_Grade_12th",
        "School_Grade_7th", "School_Grade_8th", "School_Grade_9th",
        "Phone_Usage_Purpose_Browsing", "Phone_Usage_Purpose_Education",
        "Phone_Usage_Purpose_Gaming", "Phone_Usage_Purpose_Other",
        "Phone_Usage_Purpose_Social Media","Parental_Control"
    ],
    axis=1
)


# %%
print(df_encoded.columns.tolist())

# %%
numeric.describe()
# %%

df_no_outliers = numeric.copy()
# outliers = pd.DataFrame()

for col in numeric.columns:
    q1 = numeric[col].quantile(0.25)
    q3 = numeric[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower) & (df_no_outliers[col] <= upper)]


# %%
df_no_outliers.describe()
# 




###########################
# EXPLORATORY DATA ANALYSIS

# %%

# top 5 phone usage purposes vs average time spent
sns.set(style="whitegrid", palette="pastel")


top_activities = df.groupby("Phone_Usage_Purpose")["Daily_Usage_Hours"].mean().sort_values(ascending=False).head(5)

plt.figure(figsize=(8,6))
sns.barplot(x=top_activities.index, y=top_activities.values, palette="Blues_r")
plt.title("Top 5 Activities by Avg Time Spent")
plt.ylabel("Avg Daily Usage Hours")
plt.xlabel("Phone Usage Purpose")
plt.xticks(rotation=30)
plt.show()

# usage vs academic performance
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, 
    x="Daily_Usage_Hours", 
    y="Academic_Performance", 
    hue="Addiction_Level",  
    palette="coolwarm",
    alpha=0.7
)
plt.title("Daily Usage Hours vs Academic Performance")
plt.xlabel("Daily Usage Hours")
plt.ylabel("Academic Performance")
plt.show()

#sleep patterns

plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Addiction_Level", y="Sleep_Hours", palette="Set2")

plt.title("Sleep Patterns by Addiction Level", fontsize=14, fontweight="bold")
plt.xlabel("Addiction Level")
plt.ylabel("Sleep Hours per Night")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#bedtime phone usage vs sleep quALITY

plt.figure(figsize=(8,6))
sns.boxplot(x="Screen_Time_Before_Bed", y="Sleep_Hours", data=df, palette="coolwarm")

plt.title("Impact of Bedtime Phone Usage on Sleep Hours", fontsize=14)
plt.xlabel("Phone Usage Before Bed", fontsize=12)
plt.ylabel("Sleep Hours", fontsize=12)
plt.show()


# Mental Health Indicators vs Addiction Level



# Anxiety vs Addiction
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Addiction_Level", y="Anxiety_Level", palette="Set2")
plt.title("Anxiety Level by Addiction Severity")
plt.xlabel("Addiction Level")
plt.ylabel("Anxiety Level")
plt.show()

# Depression vs Addiction
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Addiction_Level", y="Depression_Level", palette="Set2")
plt.title("Depression Level by Addiction Severity")
plt.xlabel("Addiction Level")
plt.ylabel("Depression Level")
plt.show()

# Self-Esteem vs Addiction
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Addiction_Level", y="Self_Esteem", palette="Set2")
plt.title("Self-Esteem by Addiction Severity")
plt.xlabel("Addiction Level")
plt.ylabel("Self-Esteem")
plt.show()



#%%

# DATA VISUALIZATION 

#%%
#  Daily_Usage_Hours, Sleep_Hours
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df["Daily_Usage_Hours"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Daily Usage Hours")

plt.subplot(1,2,2)
sns.histplot(df["Sleep_Hours"], bins=20, kde=True, color="salmon")
plt.title("Distribution of Sleep Hours")

plt.tight_layout()
plt.show()



#  Heatmap Correlation
plt.figure(figsize=(7,5))
corr_cols = ["Daily_Usage_Hours","Sleep_Hours","Academic_Performance","Anxiety_Level"]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Gender vs Avg Addiction_Level
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Gender", y="Addiction_Level", estimator="mean", ci=None, palette="Set2")
plt.title("Average Addiction Level by Gender")
plt.show()


# Time_on_Gaming vs Academic_Performance
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Time_on_Gaming", y="Academic_Performance", hue="Addiction_Level", palette="coolwarm")
plt.title("Gaming Time vs Academic Performance")
plt.show()



#insights from data
#%%
#Addiction vs Academic Performance 
plt.figure(figsize=(6,4))
sns.boxplot(x="Addiction_Level", y="Academic_Performance", data=df, palette="coolwarm")
plt.title("Addiction Level vs Academic Performance")
plt.show()


# Addiction vs Sleep Hours
plt.figure(figsize=(6,4))
sns.boxplot(x="Addiction_Level", y="Sleep_Hours", data=df, palette="muted")
plt.title("Impact of Addiction on Sleep Hours")
plt.show()


# Addiction vs Mental Health 
plt.figure(figsize=(8,5))
sns.lineplot(data=df, x="Addiction_Level", y="Anxiety_Level", label="Anxiety", marker="o")
sns.lineplot(data=df, x="Addiction_Level", y="Depression_Level", label="Depression", marker="o")
sns.lineplot(data=df, x="Addiction_Level", y="Self_Esteem", label="Self-Esteem", marker="o")
plt.title("Mental Health Indicators vs Addiction Level")
plt.legend()
plt.show()




# %%
X = df_no_outliers.drop(columns=["Addiction_Level"])
y = df_no_outliers["Addiction_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
# %%
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
# %%
# df_no_outliers.info()
# %%

###########################
# output of cleaned dataset
# 
# 
# df_no_outliers.to_csv("teen_phone_addiction_cleaned.csv", index=False)
# %%

