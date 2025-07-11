import pandas as pd

# Read the CSV file (assuming it's named 'data.csv')
df = pd.read_csv('Evaluation/data/adult.csv', skipinitialspace=True)

# Remove spaces from column names
df.columns = df.columns.str.replace(' ', '', regex=False)

# Remove leading/trailing spaces in all string values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Save the cleaned dataset to the same file
df.to_csv('data.csv', index=False)

print("Spaces removed and dataset saved successfully.")
