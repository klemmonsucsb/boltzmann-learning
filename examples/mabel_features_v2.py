import pandas as pd
import numpy as np

CSV_PATHS = [
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_001.csv',
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_002.csv',
]

df = pd.concat([pd.read_csv(p, low_memory=False) for p in CSV_PATHS], ignore_index=True)
print(f"Loaded {len(df)} rows")

# --- Tier 1: simple low-cardinality categoricals -> one-hot ---
simple_cat_cols = ['os', 'machine_architecture', 'endianness', 'programming_language',
                   'file_type_extension', 'binary_type']
simple_cat_cols = [c for c in simple_cat_cols if c in df.columns]
for c in simple_cat_cols:
    print(f"{c}: {df[c].nunique()} unique values")
one_hot_simple = pd.get_dummies(df[simple_cat_cols], prefix=simple_cat_cols, dummy_na=True)
print(f"Tier 1 produced {one_hot_simple.shape[1]} new columns")

# --- Tier 2: semicolon-separated list columns -> multi-hot ---
list_cols = ['yara_capabilities', 'yara_compiler_signatures', 'yara_crypto',
             'yara_anti_debug_anti_vm']
list_cols = [c for c in list_cols if c in df.columns]

multi_hot_frames = []
for c in list_cols:
    # split on ';', strip whitespace, explode into a set of tags per row
    tag_sets = df[c].fillna('').apply(lambda s: set(t.strip() for t in s.split(';') if t.strip() and t.strip() != '-'))
    all_tags = sorted(set.union(*tag_sets)) if len(tag_sets) else []
    print(f"{c}: {len(all_tags)} distinct tags found")
    if len(all_tags) > 200:
        print(f"  WARNING: {c} has {len(all_tags)} tags, likely too many to one-hot directly -- consider keeping only the most frequent ones")
    tag_matrix = pd.DataFrame(
        {f"{c}__{tag}": tag_sets.apply(lambda s: tag in s).astype(int) for tag in all_tags}
    )
    multi_hot_frames.append(tag_matrix)

multi_hot = pd.concat(multi_hot_frames, axis=1) if multi_hot_frames else pd.DataFrame(index=df.index)
print(f"Tier 2 produced {multi_hot.shape[1]} new columns")

new_features = pd.concat([one_hot_simple, multi_hot], axis=1)
new_features.to_pickle('mabel_extra_features.pkl')
print(f"\nSaved {new_features.shape[1]} new engineered features to mabel_extra_features.pkl")
