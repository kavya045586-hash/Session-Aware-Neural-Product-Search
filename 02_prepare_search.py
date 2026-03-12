import pandas as pd
import pickle

# Load your merged file
print("📂 Loading merged data to create search index...")
df = pd.read_parquet("data/electronics_merged.parquet")

# Create a Title -> ASIN mapping
# We use drop_duplicates to keep the search list clean
search_map = df[['title', 'asin']].drop_duplicates('title')

# Convert to a dictionary for lightning-fast lookups in the App
title_to_asin = dict(zip(search_map['title'], search_map['asin']))

# Save it
with open("data/title_to_asin.pkl", "wb") as f:
    pickle.dump(title_to_asin, f)

print(f"✅ Created search map with {len(title_to_asin):,} unique product names!")