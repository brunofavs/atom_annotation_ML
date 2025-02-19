import os

# Directory containing the files
directory = "sample_masks"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.startswith("mask_") and filename.endswith(".jpg"):
        new_filename = filename.replace("mask_", "", 1)  # Remove only the first occurrence
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

print("Renaming completed.")