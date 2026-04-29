import joblib
import pandas as pd
from xgboost import XGBClassifier as XGBC
from sklearn.multioutput import MultiOutputClassifier

df=pd.read_csv(r"mental_health_multilabel_dataset.csv")
feature_cols = [col for col in df.columns if not col.startswith('label_')]
label_cols = [col for col in df.columns if col.startswith('label_')]

map_education = {'Fără studii': 1, 'Școală generală': 2, 'Liceu': 3, 'Studii superioare': 4, 'Doctorat/Master avansat': 5}
map_gender = {'Female': 0, 'Male': 1, 'Other': 2}
map_job = {'Retired': 0, 'Employed': 1, 'Student': 2, 'Unemployed': 3, 'Disabled': 4}

df["gender"]=df["gender"].map(map_gender)
df["employment_status"]=df["employment_status"].map(map_job)

X_train=df[feature_cols]
Y_train=df[label_cols]
reg = MultiOutputClassifier(XGBC()).fit(X_train, Y_train)
joblib.dump(reg, "model.joblib")
joblib.dump(label_cols, "label.joblib")
joblib.dump(feature_cols, "feature.joblib")