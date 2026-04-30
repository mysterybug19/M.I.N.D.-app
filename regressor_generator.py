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

base=XGBC(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=1.0,
    scale_pos_weight=3,
    eval_metric='logloss',
    random_state=42,
)
calibrated=CalibratedClassifierCV(base, cv=5, method='isotonic')
reg=MultiOutputClassifier(calibrated).fit(X_train, Y_train)

joblib.dump(reg, "model.joblib")
joblib.dump(label_cols, "label.joblib")
joblib.dump(feature_cols, "feature.joblib")