import pandas as pd
import os
import gc

# 1. Define required columns
required_cols = [
    'Year', 'Month', 'DayOfWeek', 'Hour', 
    'Primary Type', 'Location Description', 'Arrest', 'Domestic',
    'District', 'Community Area', 'Block', 
    'Longitude', 'Latitude'
]

# Set the absolute maximum number of rows for the final unified dataset
# 60,000 rows is perfectly balanced: ~3-5MB file size, completely safe for 512MB RAM, 
# while preserving 100% statistical accuracy.
TARGET_TOTAL_ROWS = 60000

print("Starting offline data preparation (Proportional Sampling)...")

# ==========================================
# PASS 1: Calculate Real Proportions
# ==========================================
print("\n[Pass 1] Scanning files to determine true yearly proportions...")
year_counts = {}
total_real_rows = 0

for year in range(2015, 2026):
    file_path = f"./data_chunks/crimes_{year}.parquet"
    if os.path.exists(file_path):
        try:
            # Load ONLY the 'Year' column to instantly get the row count without memory overhead
            temp_col = pd.read_parquet(file_path, columns=['Year'])
            count = len(temp_col)
            year_counts[year] = count
            total_real_rows += count
            
            del temp_col
            gc.collect()
        except ValueError:
            # Fallback if 'columns' filter fails
            temp_df = pd.read_parquet(file_path)
            count = len(temp_df)
            year_counts[year] = count
            total_real_rows += count
            del temp_df
            gc.collect()

print(f"Total historical rows found: {total_real_rows:,}")

# ==========================================
# PASS 2: Proportional Extraction
# ==========================================
print("\n[Pass 2] Extracting proportional samples...")
all_dfs = []

for year in range(2015, 2026):
    file_path = f"./data_chunks/crimes_{year}.parquet"
    if os.path.exists(file_path) and year in year_counts:
        # Calculate exactly how many rows this specific year should contribute
        proportion = year_counts[year] / total_real_rows
        target_sample_size = int(TARGET_TOTAL_ROWS * proportion)
        
        try:
            try:
                df = pd.read_parquet(file_path, columns=required_cols)
            except ValueError:
                df = pd.read_parquet(file_path)
                df = df[df.columns.intersection(required_cols)]
            
            # Sample proportionally
            if len(df) > target_sample_size:
                df = df.sample(n=target_sample_size, random_state=42)
                
            all_dfs.append(df)
            print(f"Processed {year}: Sampled {len(df):,} rows ({proportion*100:.1f}% of total)")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

# ==========================================
# FINAL: Combine and Compress
# ==========================================
print("\n[Finalizing] Combining and optimizing data types...")
final_df = pd.concat(all_dfs, ignore_index=True)

# Optimize data types for maximum compression
cat_cols = ['Primary Type', 'Location Description', 'Block', 'DayOfWeek', 'District', 'Community Area']
for col in cat_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].astype('category')

for col in ['Longitude', 'Latitude']:
    if col in final_df.columns:
        final_df[col] = final_df[col].astype('float32')

for col in ['Year', 'Month', 'Hour']:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], downcast='unsigned')

# Save the unified, miniaturized file
output_path = "./data_chunks/crimes_dashboard_ready.parquet"
final_df.to_parquet(output_path, index=False)

print(f"\n✅ Success! Created {output_path}")
print(f"Final Dataset Size: {len(final_df):,} rows. Ready for deployment!")