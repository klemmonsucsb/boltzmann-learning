import pandas as pd

CSV_PATHS = [
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_001.csv',
    '/Users/kylejack/MABEL-dataset/release/malware_family/current_version/mabel_dataset_v2.10_condensed/mabel_dataset_v2.10_condensed_002.csv',
]
df = pd.concat([pd.read_csv(p, low_memory=False) for p in CSV_PATHS], ignore_index=True)

pairs_to_check = [('WinGo', 'Wingo'), ('Jaff', 'JaffRansomware')]
compare_cols = ['binary_file_size', 'entropy(min=0.0; max=8.0)', 'count_instruction_lines',
                'number_sections', 'clam_av_scan_results', 'yara_malware', 'programming_language']
compare_cols = [c for c in compare_cols if c in df.columns]

for a, b in pairs_to_check:
    print(f"\n=== Comparing '{a}' vs '{b}' ===")
    group_a = df[df['family_name'] == a]
    group_b = df[df['family_name'] == b]
    print(f"  '{a}': {len(group_a)} samples")
    print(f"  '{b}': {len(group_b)} samples")
    for col in compare_cols:
        if df[col].dtype in ['float64', 'int64']:
            print(f"  {col}: {a} mean={group_a[col].mean():.2f}, {b} mean={group_b[col].mean():.2f}")
        else:
            print(f"  {col}: {a} values={group_a[col].unique()[:3]}, {b} values={group_b[col].unique()[:3]}")

# Also check Msil/NSIS more carefully -- are these coherent groups or scattered catch-alls?
print("\n=== Checking Msil and NSIS for internal consistency ===")
for name in ['Msil', 'NSIS']:
    if name in df['family_name'].values:
        group = df[df['family_name'] == name]
        print(f"\n'{name}': {len(group)} samples")
        if 'clam_av_scan_results' in df.columns:
            print(f"  clamav results: {group['clam_av_scan_results'].unique()[:5]}")
        if 'entropy(min=0.0; max=8.0)' in df.columns:
            print(f"  entropy range: {group['entropy(min=0.0; max=8.0)'].min():.2f} to {group['entropy(min=0.0; max=8.0)'].max():.2f}")
