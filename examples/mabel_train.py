import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

CSV_PATHS = [
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_001.csv',
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_002.csv',
]
MIN_SAMPLES = 100

print("Loading CSVs...")
df = pd.concat([pd.read_csv(p, low_memory=False) for p in CSV_PATHS], ignore_index=True)
print(f"Loaded {len(df)} total rows")

counts = df['family_name'].value_counts()
keep_families = counts[counts >= MIN_SAMPLES].index
df = df[df['family_name'].isin(keep_families)].reset_index(drop=True)
print(f"Kept {len(df)} rows across {len(keep_families)} families (>= {MIN_SAMPLES} samples each)")

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
print(f"Using {X.shape[1]} numeric/boolean features")

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.15, random_state=1, stratify=y)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

params_paper = {"boosting": "gbdt", "objective": "multiclass", "num_class": num_classes,
                "num_iterations": 1000, "learning_rate": 0.05, "num_leaves": 2048,
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
    predictions.append(preds)
    print(f"  done, prediction array shape: {preds.shape}")

with open('mabel_predictions.pkl', 'wb') as f:
    pickle.dump({
        'predictions': predictions,
        'y_test': y_test,
        'classes': le.classes_,
    }, f)
print("Saved mabel_predictions.pkl")
