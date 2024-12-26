import pandas as pd

# Load original dataset
df = pd.read_csv("downsampled_dataset.csv")

# Randomly sample 2% of the dataset
df_downsampled = df.sample(frac=0.02, random_state=42)

# Save the downsampled dataset to a new CSV file
df_downsampled.to_csv("sample.csv", index=False)
