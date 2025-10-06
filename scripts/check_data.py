import pandas as pd
import numpy as np

# Read only first 1000 rows for quick analysis
df = pd.read_csv(
    "G:/spatial_data/processed/20230523_HCC_PRISM_probe_refined/readout/intensity_standard.csv",
    nrows=1000,
)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check for NaN values
print("NaN values per column:")
for col in df.columns:
    nan_count = df[col].isna().sum()
    print(f"{col}: {nan_count}")

# Check sum calculation
df["sum"] = df["ch1"] + df["ch2"] + df["ch4"]
zero_sum = (df["sum"] == 0).sum()
print(f"Sum = 0: {zero_sum}")
print(f"Sum min: {df['sum'].min()}")
print(f"Sum max: {df['sum'].max()}")

# Check ratios
df["ch1/A"] = df["ch1"] / df["sum"]
df["ch2/A"] = df["ch2"] / df["sum"]
df["ch3/A"] = df["ch3"] / df["sum"]
df["ch4/A"] = df["ch4"] / df["sum"]

print("Ratio statistics:")
for ratio in ["ch1/A", "ch2/A", "ch3/A", "ch4/A"]:
    nan_count = df[ratio].isna().sum()
    inf_count = np.isinf(df[ratio]).sum()
    print(f"{ratio}: NaN={nan_count}, Inf={inf_count}")


