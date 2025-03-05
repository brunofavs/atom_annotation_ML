import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Read the CSV file
file_path = 'train_info_unet_200.csv'
df = pd.read_csv(file_path)

# Identify columns to smooth (excluding 'Epochs')
columns_to_smooth = ['Train losses', 'Validation losses', 'Train DICE', 'Validation DICE']

# Apply Savitzky-Golay filter
window_length = 5  # Choose an odd number; adjust based on your data's noise level
poly_order = 2     # Polynomial order, typically less than window_length

for col in columns_to_smooth:
    df[col] = savgol_filter(df[col], window_length=window_length, polyorder=poly_order)


# Save the smoothed data to a new CSV file
output_path = 'train_info_unet_200_smooth.csv'
df.to_csv(output_path, index=False)

print("Smoothing completed and data saved to:", output_path)