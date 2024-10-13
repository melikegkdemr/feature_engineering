import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets\diabetes.csv")
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Describes #####################")
    print(dataframe.describe().T)

check_df(df)

def grab_col_names(dff, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dff.columns if dff[col].dtypes == "O"]
    num_but_cat = [col for col in dff.columns if dff[col].nunique() < cat_th and
                   dff[col].dtypes != "O"]
    cat_but_car = [col for col in dff.columns if dff[col].nunique() > car_th and
                   dff[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dff.columns if dff[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("Observations", dff.shape[0])
    print("Variables", dff.shape[1])
    print("cat_cols", len(cat_cols))
    print("num_cols", len(num_cols))
    print("cat_but_car", len(cat_but_car))
    print("num_but_cat", len(num_but_cat))

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols

df["Outcome"].value_counts()
df["Outcome"].value_counts() / len(df) * 100

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome")

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True) # buradan şu sonucu çıkartıyorum: demek ki burada null olan değerlere 0 değeri girilmiş olabilir.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# AYKIRI GÖZLEM ANALİZİ - OUTLİERS

# 1) q1=0.1 ve q3=0.9 / model accuracy: 

def outlier_thresholds(df, col_name, q1=0.1, q3=0.9):
    q1 = df[col_name].quantile(q1)
    q3 = df[col_name].quantile(q3)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit

outlier_thresholds(df,"Pregnancies")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def box_plot_analyses(df, col):
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()

for col in num_cols:
    print(col)
    box_plot_analyses(df,col)

# EKSİK GÖZLEM ANALİZİ - MISSING VALUES

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")  # bir boşluk bırakmak için \n koyuyoruz

    if na_name:
        return na_columns


missing_values_table(df, True)

##################################
# KORELASYON
##################################

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

##################################
# BASE MODEL KURULUMU
##################################

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model_1 = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model_1.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model_1, X)

# FEATURE ENGINEERING

selected=["Glucose","SkinThickness","Insulin","BMI","BloodPressure"]

for col in selected:
    df[col]= df[col].apply(lambda x: np.nan if x == 0 else x)

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")  # bir boşluk bırakmak için \n koyuyoruz

    if na_name:
        return na_columns


missing_values_table(df, True)

# eksik değerleri outcome kırılımda dolduralım 
for col in selected:
    df[col]=df[col].fillna(df.groupby("Outcome")[col].transform("mean"))

df.isnull().sum()

### OUTLIER DETECTION

def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# with these up and low values , we are gonna look at the variables if there is a outlier or not 
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

df.columns= [col.upper() for col in df.columns]
df.columns

##################################
# ÖZELLİK ÇIKARIMI
##################################

df["AGE_NEW"]= pd.cut(df["AGE"],bins= [20,45,max(df["AGE"])], labels=["mature","senior"])

df["GLUCOSE_NEW"]= pd.cut(df["GLUCOSE"], bins=[0, 100, 140 , max(df["GLUCOSE"])], labels=["low","normal","high"])

df["BMI_NEW"]=pd.cut(df["BMI"], bins=[18,25,32,max(df["BMI"])], labels=["Normal Weight","Overweight","Obese"])

df.loc[df["INSULIN"]<=130,"INSULIN_NEW"]="normal"
df.loc[df["INSULIN"]>130, "INSULIN_NEW"]="anormal"

df["GLUCOSE_INSULIN"]=df["GLUCOSE"]*df["INSULIN"]
df["INSULIN_BMI"]=df["INSULIN"]*df["BMI"]
df["GLUCOSE_BLOODPRESSURE"]= df["GLUCOSE"]* df["BLOODPRESSURE"]
df["INSULIN_BLOODPRESSURE"]= df["INSULIN"]*df["BLOODPRESSURE"]

# AGE-GLUCOSE
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="maturelow"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="maturenormal"
df.loc[(df["AGE_NEW"]=="mature") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="maturehigh"

df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="low"),"AGE_GLUCOSE"]="seniorlow"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="normal"),"AGE_GLUCOSE"]="seniornormal"
df.loc[(df["AGE_NEW"]=="senior") & (df["GLUCOSE_NEW"]=="high"),"AGE_GLUCOSE"]="seniorhigh"

# AGE-BMI
df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Normal Weight"),"AGE_BMI"]="matureNormalWeight"
df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Overweight"),"AGE_BMI"]="matureOverweight"
df.loc[(df["AGE_NEW"]=="mature") & (df["BMI_NEW"]=="Obese"),"AGE_BMI"]="matureObese"

df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Normal Weight"),"AGE_BMI"]="seniorNormalWeight"
df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Overweight"),"AGE_BMI"]="seniorOverweight"
df.loc[(df["AGE_NEW"]=="senior") & (df["BMI_NEW"]=="Obese"),"AGE_BMI"]="seniorObese"

##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols

cat_cols=[ col for col in cat_cols if col != "OUTCOME"]

from sklearn.preprocessing import LabelEncoder

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df,col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi 

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols]

df = one_hot_encoder(df, cat_cols, drop_first=True)

# STANDARTLAŞTIRMA

from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()

# MODELLEME

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


from lightgbm import LGBMClassifier

lgbm_model= LGBMClassifier(random_state=42, verbosity=-1)
lgbm_model.fit(X_train, y_train)
y_pred_2 =lgbm_model.predict(X_test)
lgbm_accuracy= accuracy_score(y_pred_2, y_test) 
lgbm_accuracy

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_3 = log_model.predict(X_test)
logistic_sonuc = accuracy_score(y_pred_3, y_test)
logistic_sonuc

from xgboost import XGBClassifier

xgm_model= XGBClassifier(random_state=42)
xgm_model.fit(X_train, y_train)
y_pred_4 =xgm_model.predict(X_test)
xgb= accuracy_score(y_pred_4, y_test)  
xgb

from sklearn.tree import DecisionTreeClassifier

des_model = DecisionTreeClassifier(random_state=42)
des_model.fit(X_train, y_train)
y_pred_2 = des_model.predict(X_test)
decison_sonuc = accuracy_score(y_pred_2, y_test)
decison_sonuc

from sklearn.neighbors import KNeighborsClassifier

knn_model= KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_6 =knn_model.predict(X_test)
knn= accuracy_score(y_pred_6, y_test) # 0.84
knn

models= [rf_model,lgbm_model,des_model,log_model,xgm_model,knn_model]

best_model = None
best_accuracy = 0

for i, model in enumerate(models,1):
    model.fit(X_train, y_train)
    y_pred_i= model.predict(X_test)
    accuracy_score_model=accuracy_score(y_pred_i, y_test)
    
    print(f'Model Name: {type(model).__name__}, Accuracy: {accuracy_score_model}\n')
    
    print("#"*80)
    
    if accuracy_score_model> best_accuracy:
        best_accuracy=accuracy_score_model
        best_model = model
        
print(f"Best Model {best_model}, Best Accuracy {best_accuracy}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_model, X_train)