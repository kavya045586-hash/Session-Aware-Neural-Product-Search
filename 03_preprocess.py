import pandas as pd
import pickle
import numpy as np

# Load the merged data you inspected earlier
print("📂 Loading 14.4M interactions...")
df = pd.read_parquet("data/electronics_merged.parquet")

# 1. Create the Item Mapper (ASIN to Integer ID)
unique_asins = df['asin'].unique().tolist()
with open("data/item_mapper.pkl", "wb") as f:
    pickle.dump(unique_asins, f)

asin_to_idx = {asin: idx for idx, asin in enumerate(unique_asins)}

# 2. Convert sessions to sequences (Last 5 items)
print("🔄 Creating user sequences...")
df = df.sort_values(['user_id', 'timestamp'])
sequences = df.groupby('user_id')['asin'].apply(list)

processed_data = []
for seq in sequences:
    if len(seq) >= 2:
        # Convert ASINs to indices
        idx_seq = [asin_to_idx[item] for item in seq]
        # Sliding window of 6 (5 inputs + 1 target)
        for i in range(len(idx_seq) - 5):
            processed_data.append(idx_seq[i:i+6])

# 3. Save for training
seq_df = pd.DataFrame(processed_data)
seq_df.to_parquet("data/user_sequences.parquet")
print(f"✅ Preprocessing complete. Generated {len(seq_df):,} sequences.")