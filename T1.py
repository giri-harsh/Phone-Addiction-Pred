#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt
import seaborn as sns

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
