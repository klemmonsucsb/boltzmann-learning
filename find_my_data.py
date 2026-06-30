import os

print("Searching for saved data files inside the project...")
found_any = False

# Walk through all directories starting from the current location
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pkl') or 'rps_40000' in root:
            full_path = os.path.join(root, file)
            print(f"FOUND: {full_path}")
            found_any = True

if not found_any:
    print("Could not find any simulation files anywhere in this directory tree.")
