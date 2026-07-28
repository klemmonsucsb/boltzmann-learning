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

df = pd.concat([pd.read_csv(p, low_memory=False) for p in CSV_PATHS], ignore_index=True)
counts = df['family_name'].value_counts()
keep_families = counts[counts >= MIN_SAMPLES].index
df = df[df['family_name'].isin(keep_families)].reset_index(drop=True)
y_raw = df['family_name']
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)
drop_cols = ['family_name', 'sha256_hash', 'md5_hash', 'sha1_hash', 'sha224_hash',
             'sha384_hash', 'sha512_hash', 'sample_name', 'ssdeep', 'imphash']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
bool_like = [c for c in X.columns if X[c].dropna().isin(['TRUE', 'FALSE']).all()]
for c in bool_like:
    X[c] = (X[c] == 'TRUE').astype(int)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols + bool_like].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.15, random_state=1, stratify=y)

# Original (500 iter, 50 leaves) vs new (1500 iter, 150 leaves) -- WITH actual
# validation tracking this time, so we can see what's really happening instead
# of guessing.
for label, params in [
    ("original", {"boosting": "gbdt", "objective": "multiclass", "num_class": num_classes,
                  "num_iterations": 500, "learning_rate": 0.05, "num_leaves": 50,
                  "max_depth": 10, "min_data_in_leaf": 20, "feature_fraction": 0.5, "verbose": -1}),
    ("bigger", {"boosting": "gbdt", "objective": "multiclass", "num_class": num_classes,
                "num_iterations": 300, "learning_rate": 0.02, "num_leaves": 63,
                "max_depth": 10, "min_data_in_leaf": 20, "feature_fraction": 0.5, "verbose": -1}),
]:
    print(f"\n=== {label} ===")
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_test, y_test, reference=train_set)
    evals_result = {}
    model = lgb.train(params, train_set, valid_sets=[train_set, valid_set],
                      valid_names=['train', 'test'],
                      callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(period=100)])
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, np.argmax(preds, axis=1))
    print(f"{label}: final test accuracy = {acc:.4f}")
    print(f"  train multi_logloss at end: {evals_result['train']['multi_logloss'][-1]:.4f}")
    print(f"  test multi_logloss at end: {evals_result['test']['multi_logloss'][-1]:.4f}")
    print(f"  test multi_logloss at start (round 100): {evals_result['test']['multi_logloss'][0]:.4f}")
