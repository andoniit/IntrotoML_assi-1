import pandas as pd

# Read the sorted CSV file
file_path = 'lab01_dataset_1.csv'
df = pd.read_csv(file_path)

# Sort the DataFrame by the 'Score' column in ascending order
df.sort_values(by='Score', inplace=True)

# Define the threshold values
threshold_46 = 46.0
threshold_69_5 = 69.5
threshold_81_5 = 81.5

# Create new columns with boolean values
df['Score_46.0'] = df['Score'] < threshold_46
df['Score_69.5'] = df['Score'] < threshold_69_5
df['Score_81.5'] = df['Score'] < threshold_81_5

# Convert boolean values to True/False strings
df[['Score_46.0', 'Score_69.5', 'Score_81.5']] = df[['Score_46.0', 'Score_69.5', 'Score_81.5']].astype(bool)

# Drop the original "Score" column
df = df.drop('Score', axis=1)

# Reorder columns, moving "Output" to the last position
column_order = [col for col in df.columns if col != 'Output'] + ['Output']
df = df[column_order]

# Check for missing values
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"Dataset has {missing_values} missing values.")
    # You can choose to handle missing values as needed.

# Check for redundant or repeated input samples
duplicate_rows = df[df.duplicated()]
if not duplicate_rows.empty:
    print("Dataset has redundant or repeated input samples.")
    
    # Print the duplicate rows
    print("Duplicate rows:")
    print(duplicate_rows)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

# Check for contradicting <input, output> pairs
contradicting_rows = df[df.duplicated(subset=['Mood', 'Effort', 'Score_46.0', 'Score_69.5', 'Score_81.5', 'Output'], keep=False)]
if not contradicting_rows.empty:
    print("Dataset has contradicting <input, output> pairs.")
    df = df.drop_duplicates(subset=['Mood', 'Effort', 'Score_46.0', 'Score_69.5', 'Score_81.5', 'Output'])
    print("Contradicting rows removed.")

# Save the modified and cleaned dataset to a new CSV file
output_file_path = 'lab01_dataset_1_updated.csv'
df.to_csv(output_file_path, index=False)

print(f"Modified and cleaned dataset saved to {output_file_path}.")
