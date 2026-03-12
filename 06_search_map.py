import pandas as pd
import pickle
import os

# 1. Load the Item Mapper (to ensure we match the model's indices)
print("📂 Loading item mapper...")
with open("data/item_mapper.pkl", "rb") as f:
    unique_asins = pickle.load(f)

# Create a fast lookup: ASIN -> Index
asin_to_idx = {asin: idx for idx, asin in enumerate(unique_asins)}

# 2. Load the Merged Data
# We only need 'title' and 'asin' to create the search bar dictionary
print("📖 Reading product titles from merged file...")
df = pd.read_parquet("data/electronics_merged.parquet", columns=['title', 'asin'])

# 3. Create Title -> Index Mapping
print("🛠️ Building search map...")
# Remove duplicates so the search bar isn't cluttered
search_df = df.drop_duplicates(subset=['title'])

title_to_idx = {}
for _, row in search_df.iterrows():
    title = str(row['title'])
    asin = row['asin']
    
    # Only add to map if the item exists in our trained model
    if asin in asin_to_idx:
        # We limit title length to 150 chars for better UI display
        title_to_idx[title[:150]] = asin_to_idx[asin]

# 4. Save the Search Map
print(f"💾 Saving map with {len(title_to_idx):,} searchable titles...")
with open("data/title_to_idx.pkl", "wb") as f:
    pickle.dump(title_to_idx, f)

print("✅ Success! Your search index is ready for Step 7.")