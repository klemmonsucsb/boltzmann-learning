import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_PATHS = [
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_001.csv',
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_002.csv',
]
MIN_SAMPLES = 100

print("Loading base data...")
df = pd.concat([pd.read_csv(p, low_memory=False) for p in CSV_PATHS], ignore_index=True)

print("Loading engineered features...")
extra_features = pd.read_pickle('mabel_extra_features.pkl')
assert len(extra_features) == len(df), "Row count mismatch -- extra features don't align with base data!"

counts = df['family_name'].value_counts()
keep_families = counts[counts >= MIN_SAMPLES].index
mask = df['family_name'].isin(keep_families)
df = df[mask].reset_index(drop=True)
extra_features = extra_features[mask.values].reset_index(drop=True)
print(f"Kept {len(df)} rows across {len(keep_families)} families")

y_raw = df['family_name']
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

drop_cols = ['family_name', 'sha256_hash', 'md5_hash', 'sha1_hash', 'sha224_hash',
             'sha384_hash', 'sha512_hash', 'sample_name', 'ssdeep', 'imphash']
X_base = df.drop(columns=[c for c in drop_cols if c in df.columns])
bool_like = [c for c in X_base.columns if X_base[c].dropna().isin(['TRUE', 'FALSE']).all()]
for c in bool_like:
    X_base[c] = (X_base[c] == 'TRUE').astype(int)
numeric_cols = X_base.select_dtypes(include=[np.number]).columns.tolist()
X_base = X_base[numeric_cols + bool_like].fillna(0)

X_combined = pd.concat([X_base.reset_index(drop=True), extra_features.reset_index(drop=True)], axis=1).fillna(0)
print(f"Combined feature count: {X_combined.shape[1]} (was {X_base.shape[1]} before engineering)")

X_train, X_test, y_train, y_test = train_test_split(X_combined.values, y, test_size=0.15, random_state=1, stratify=y)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

params_paper = {"boosting": "gbdt", "objective": "multiclass", "num_class": num_classes,
                "num_iterations": 800, "learning_rate": 0.03, "num_leaves": 512,
                "max_depth": 15, "min_data_in_leaf": 50, "feature_fraction": 0.5, "verbose": -1}
params_shallow = {"boosting": "gbdt", "objective": "multiclass", "num_class": num_classes,
                  "num_iterations": 300, "learning_rate": 0.02, "num_leaves": 63,
                  "max_depth": 10, "min_data_in_leaf": 20, "feature_fraction": 0.5, "verbose": -1}
params_rf = {"boosting": "rf", "objective": "multiclass", "num_class": num_classes,
             "num_iterations": 1000, "learning_rate": 0.05, "num_leaves": 10,
             "max_depth": 5, "min_data_in_leaf": 10, "feature_fraction": 0.5,
             "bagging_freq": 1, "bagging_fraction": 0.8, "verbose": -1}

predictions = []
for name, params in [("paper", params_paper), ("shallow", params_shallow), ("rf", params_rf)]:
    print(f"Training {name} classifier...")
    train_set = lgb.Dataset(X_train, y_train)
    model = lgb.train(params, train_set)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, np.argmax(preds, axis=1))
    print(f"  {name}: accuracy = {acc:.4f}")
    predictions.append(preds)

with open('mabel_predictions_v2.pkl', 'wb') as f:
    pickle.dump({'predictions': predictions, 'y_test': y_test, 'classes': le.classes_}, f)
print("Saved mabel_predictions_v2.pkl")
