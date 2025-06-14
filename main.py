import pandas as pd
import os
from prophet import Prophet # Import Prophet at the top
import matplotlib.pyplot as plt # Import plotting libs at the top
import seaborn as sns # Import plotting libs at the top
import joblib

# Define the path to your dataset
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
print("Unique values in MONAT column before cleaning:", df_specific['MONAT'].unique()[:20]) # Show more for context


# --- REVISED CLEANING STEP FOR 'MONAT' COLUMN ---
print("\nCleaning 'MONAT' column: Extracting month number and removing non-month entries...")

# 1. Filter out 'Summe' first (as it's a string, not a month)
df_specific = df_specific[df_specific['MONAT'] != 'Summe'].copy()

# 2. Extract the last two digits from the 'MONAT' column, which represent the actual month
# Example: '202001' becomes '01', '201912' becomes '12'
# Use errors='coerce' to turn anything that isn't a string with 2 digits at the end into NaN
df_specific['MONAT_EXTRACTED'] = df_specific['MONAT'].astype(str).str[-2:]

# 3. Convert the extracted month to numeric, coercing errors to NaN
df_specific['MONAT_NUM'] = pd.to_numeric(df_specific['MONAT_EXTRACTED'], errors='coerce')

# 4. Drop rows where MONAT_NUM is NaN (failed to convert to a number)
df_specific.dropna(subset=['MONAT_NUM'], inplace=True)

# 5. Ensure month is within 1-12 range
df_specific['MONAT_NUM'] = df_specific['MONAT_NUM'].astype(int)
df_specific = df_specific[(df_specific['MONAT_NUM'] >= 1) & (df_specific['MONAT_NUM'] <= 12)].copy()


print(f"Rows after cleaning and filtering 'MONAT' column: {len(df_specific)}")
print("First 5 rows after month cleaning:")
print(df_specific.head())
print("Unique values in MONAT_NUM column after cleaning:", df_specific['MONAT_NUM'].unique())


# --- Prepare 'Date' column for time series analysis ---
print("\nPreparing 'Date' column for time series analysis...")

# Pad 'MONAT_NUM' with leading zeros for date formatting
df_specific['MONAT_PADDED'] = df_specific['MONAT_NUM'].astype(str).str.zfill(2)

df_specific['Date'] = pd.to_datetime(
    df_specific['JAHR'].astype(str) + '-' + df_specific['MONAT_PADDED'] + '-01',
    format='%Y-%m-%d' # Specify the format explicitly for robustness
)
df_specific.set_index('Date', inplace=True)
df_specific.sort_index(inplace=True)

print("\nSpecific data with Date index (head and tail to check range):")
print(df_specific.head())
print(df_specific.tail())

# Select the 'WERT' column as the target for prediction
df_model_data = df_specific[['WERT']].copy()

print("\nData ready for modeling (first 5 rows of 'WERT'):")
print(df_model_data.head())
print(df_model_data.info())

print("\nData preparation complete. 'df_model_data' is ready for modeling.")


# --- AI Model Creation and Forecasting ---

# Prophet requires columns named 'ds' (datetime) and 'y' (value)
df_prophet_format = df_model_data.reset_index().rename(columns={'Date': 'ds', 'WERT': 'y'})

# Ensure 'ds' is datetime type (already should be, but good check)
df_prophet_format['ds'] = pd.to_datetime(df_prophet_format['ds'])

# Convert 'y' (WERT) to numeric type and drop NaNs (already done in df_model_data, but defensive)
df_prophet_format['y'] = pd.to_numeric(df_prophet_format['y'], errors='coerce')
df_prophet_format.dropna(subset=['y'], inplace=True)


# --- DEBUGGING BLOCK (KEEP THIS FOR NOW) ---
print("\n--- DEBUG: df_prophet_format before model training (Final Check) ---")
print("Is DataFrame empty?", df_prophet_format.empty)
print("Number of rows (len):", len(df_prophet_format))
print("Info (check non-null count for 'ds' and 'y'):")
df_prophet_format.info()
print("Head (first 5 rows):")
print(df_prophet_format.head())
print("Tail (last 5 rows):")
print(df_prophet_format.tail())
print("Unique values in 'y' column (first 10):", df_prophet_format['y'].unique()[:10])
print("Sum of NaN values in 'ds':", df_prophet_format['ds'].isnull().sum())
print("Sum of NaN values in 'y':", df_prophet_format['y'].isnull().sum())
print("---------------------------------------------------\n")
# --- END DEBUGGING BLOCK ---


# --- 1. Initialize and Train the Prophet Model ---
print("\nInitializing and training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(df_prophet_format)
print("Model training complete.")

# --- 2. Make Future Predictions ---
print("\nMaking prediction for January 2021...")
future = model.make_future_dataframe(periods=1, freq='MS', include_history=False)
forecast = model.predict(future)
predicted_value_2021_01 = forecast['yhat'].iloc[0]
print(f"Predicted number of 'Alkoholunfälle' for 2021-01 (insgesamt): {predicted_value_2021_01:.2f}")

# --- 3. Visualize Historically and Add Prediction ---
print("\nCreating historical visualization with prediction...")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_prophet_format, x='ds', y='y', marker='o', label='Historical Accidents (Up to 2020)')
plt.plot(forecast['ds'].iloc[0], predicted_value_2021_01, marker='X', color='red', markersize=10, label='Predicted 2021-01')

plt.title('Historical Alcohol-Related Accidents (Munich) with 2021-01 Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Accidents (WERT)')
plt.grid(True)
plt.legend()
plt.tight_layout()

image_filename = 'historical_accidents_forecast.png'
plt.savefig(image_filename)
print(f"Visualization saved as '{image_filename}'")
plt.show()

print("\nAI Model creation, forecasting, and visualization complete.")
print("The script has predicted the value and saved the plot.")

model_filename = 'prophet_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained Prophet model saved as '{model_filename}'")

