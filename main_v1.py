import pandas as pd
import os

file_path = 'monatszahlen_verkehrsunfaelle.csv'

print(f"Loading data from: {file_path}")
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() # Clean column names

    print("Data loaded successfully!")
    print("First 5 rows:")
    print(df.head())

    print("\n--- DEBUG: ALL COLUMNS IN DATAFRAME (After stripping whitespace) ---")
    for col in df.columns:
        print(f"'{col}' (Length: {len(col)})")
    print("--- END DEBUG ---")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure it's in the correct directory.")
    exit()

print("\nFiltering data to include records up to and including 2020...")
df_filtered = df[df['JAHR'] <= 2020].copy()
print(f"Original rows: {len(df)}, Rows after filtering: {len(df_filtered)}")

print("\nFiltering for 'Alkoholunfälle' and 'insgesamt'...")
df_specific = df_filtered[
    (df_filtered['MONATSZAHL'] == 'Alkoholunfälle') &
    (df_filtered['AUSPRAEGUNG'] == 'insgesamt')
].copy()

print(f"Rows for specific category/type: {len(df_specific)}")
print("First 5 rows of specific data before month cleaning:")
print(df_specific.head())
print("Unique values in MONAT column before cleaning:", df_specific['MONAT'].unique()) # Check this output!

# --- CRITICAL NEW/REFINED FILTERING STEP FOR 'MONAT' COLUMN ---
print("\nCleaning 'MONAT' column: Converting to numeric and removing non-month entries...")

# Convert 'MONAT' column to numeric, coercing any errors (like 'Summe') to NaN
df_specific['MONAT_NUM'] = pd.to_numeric(df_specific['MONAT'], errors='coerce')

# Drop rows where 'MONAT_NUM' is NaN (meaning the original 'MONAT' was not a number, e.g., 'Summe')
df_specific.dropna(subset=['MONAT_NUM'], inplace=True)

# Now, convert 'MONAT_NUM' to integer, as it's guaranteed to be numeric
df_specific['MONAT_NUM'] = df_specific['MONAT_NUM'].astype(int)

# Optional: Further filter to ensure month is between 1 and 12, though 'Summe' is the primary issue
df_specific = df_specific[(df_specific['MONAT_NUM'] >= 1) & (df_specific['MONAT_NUM'] <= 12)].copy()


print(f"Rows after cleaning and filtering 'MONAT' column: {len(df_specific)}")
print("First 5 rows after month cleaning:")
print(df_specific.head())
print("Unique values in MONAT_NUM column after cleaning:", df_specific['MONAT_NUM'].unique()) # Should only be numbers 1-12


# --- Prepare for Time Series Analysis (combine JAHR and MONAT_NUM) ---
print("\nPreparing 'Date' column for time series analysis...")

# Pad 'MONAT_NUM' with leading zeros
df_specific['MONAT_PADDED'] = df_specific['MONAT_NUM'].astype(str).str.zfill(2)

df_specific['Date'] = pd.to_datetime(
    df_specific['JAHR'].astype(str) + '-' + df_specific['MONAT_PADDED'] + '-01',
    format='%Y-%m-%d' # Specify the format explicitly for robustness
)
df_specific.set_index('Date', inplace=True)
df_specific.sort_index(inplace=True)

print("\nSpecific data with Date index (head and tail to check range):")
print(df_specific.head())
print(df_specific.tail()) # This should show dates up to 2020-12-01

# Select the 'WERT' column as the target for prediction
df_model_data = df_specific[['WERT']].copy()

print("\nData ready for modeling (first 5 rows of 'WERT'):")
print(df_model_data.head())
print(df_model_data.info())

print("\nData preparation complete. 'df_model_data' is ready for modeling.")