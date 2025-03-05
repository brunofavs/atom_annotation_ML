import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('train_info_unet_200.csv')

# Extract the 'Validation DICE' column
validation_dice = df['Validation DICE']

# Find the maximum value in the validation dice column
max_dice = validation_dice.max()

# Find the index of the first occurrence of the maximum value
max_index = validation_dice.idxmax()

# Get the corresponding epoch number
epoch_at_max = df.loc[max_index, 'Epochs']

# Output the result
print(f"The maximum validation DICE value is {max_dice} and it occurs at epoch {epoch_at_max}.")