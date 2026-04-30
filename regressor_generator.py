import joblib
import pandas as pd
from xgboost import XGBClassifier as XGBC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

df=pd.read_csv(r"mental_health_multilabel_dataset.csv")
feature_cols = [col for col in df.columns if not col.startswith('label_')]
label_cols = [col for col in df.columns if col.startswith('label_')]

map_gender = {'Female': 0, 'Male': 1, 'Other': 2}
map_job = {'Retired': 0, 'Employed': 1, 'Student': 2, 'Unemployed': 3, 'Disabled': 4}

df["gender"]=df["gender"].map(map_gender)
df["employment_status"]=df["employment_status"].map(map_job)

X_train=df[feature_cols]
Y_train=df[label_cols]

models = {}

for col in label_cols:
    pos = Y_train[col].sum()
    neg = len(Y_train) - pos
    weight = neg / max(pos, 1)

    model = XGBC(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=weight,
        random_state=42
    )

    model.fit(X_train, Y_train[col])
    models[col] = model

joblib.dump(models, "model.joblib")
joblib.dump(label_cols, "label.joblib")
joblib.dump(feature_cols, "feature.joblib")