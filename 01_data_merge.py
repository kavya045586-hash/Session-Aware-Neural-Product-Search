import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gzip, json, os, re, time
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
META_FILE     = DATA_DIR / "meta_Electronics.jsonl.gz"
REVIEW_FILE   = DATA_DIR / "Electronics.jsonl.gz"
OUT_MERGED    = DATA_DIR / "electronics_merged.parquet"
OUT_DIR       = DATA_DIR / "processed"

SAMPLING_RATE = 0.33 
CHUNK_SIZE    = 500_000

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CATEGORY RESCUE ──────────────────────────────────────────────────────────
CATEGORY_RULES = [
    (r"\b(iphone|galaxy|pixel|smartphone|android phone)\b", "Cell Phones"),
    (r"\b(ipad|tablet|kindle|fire hd)\b", "Tablets"),
    (r"\b(macbook|laptop|notebook|chromebook)\b", "Laptops"),
    (r"\b(headphone|earbud|earphone|airpod)\b", "Audio"),
    (r"\b(camera|dslr|mirrorless|lens)\b", "Cameras"),
    (r"\b(gaming|playstation|xbox|nintendo)\b", "Gaming"),
]

def infer_category(title: str, current_cat: str) -> str:
    t = str(title).lower()
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, t): return cat
    return current_cat

# ─── STEP 1: LOAD METADATA ───────────────────────────────────────────────────
print("Step 1: Building Product Catalog...")
meta_rows = []
with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            # Master key is parent_asin (or asin if parent is missing)
            asin = item.get("parent_asin") or item.get("asin")
            if not asin: continue
            
            title = str(item.get("title", ""))
            cat = item.get("main_category", "Electronics")
            
            meta_rows.append({
                "asin": str(asin).strip(),
                "title": title[:200],
                "main_category": infer_category(title, cat),
                "brand": str(item.get("store", "Unknown"))[:100]
            })
        except: continue

meta_df = pd.DataFrame(meta_rows).drop_duplicates("asin").set_index("asin")
print(f"  ✓ Catalog ready: {len(meta_df):,} items.")

# ─── STEP 2: MERGE REVIEWS ────────────────────────────────────────────────────
print(f"Step 2: Merging reviews (Sampling {SAMPLING_RATE*100}%)...")
writer = None
total_rows = 0
t_start = time.time()

reader = pd.read_json(str(REVIEW_FILE), lines=True, compression="gzip", chunksize=CHUNK_SIZE)

for chunk in reader:
    # 1. Random Sample
    chunk = chunk.sample(frac=SAMPLING_RATE, random_state=42)

    # 2. Key Selection
    # If the review has parent_asin, we use that as our ID
    chunk["join_key"] = chunk.get("parent_asin", chunk.get("asin"))
    
    # IMPORTANT: Remove 'asin' and 'parent_asin' from chunk before joining 
    # to prevent the "Duplicate Column" error
    cols_to_drop = [c for c in ["asin", "parent_asin", "title"] if c in chunk.columns]
    chunk = chunk.drop(columns=cols_to_drop)

    # 3. Join with Metadata
    chunk["join_key"] = chunk["join_key"].astype(str).str.strip()
    chunk = chunk.join(meta_df, on="join_key", how="inner")

    if chunk.empty:
        continue

    # 4. Standardize Columns
    if "overall" in chunk.columns: chunk = chunk.rename(columns={"overall": "rating"})
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    
    # 5. Fix Timestamps (normalize seconds to milliseconds)
    mask_sec = chunk["timestamp"] < 10**11
    chunk.loc[mask_sec, "timestamp"] = chunk.loc[mask_sec, "timestamp"] * 1000
    
    # 6. Final Clean: Rename join_key back to asin for the final file
    chunk = chunk.rename(columns={"join_key": "asin"})
    chunk = chunk.drop_duplicates(subset=["user_id", "asin", "timestamp"])

    # 7. Safety Check: Remove any lingering duplicates
    chunk = chunk.loc[:, ~chunk.columns.duplicated()]

    # 8. Save to Parquet
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(OUT_MERGED), table.schema)
    writer.write_table(table)

    total_rows += len(chunk)
    print(f"  ✓ Processed {total_rows:,} rows... ({(time.time()-t_start)/60:.1f} min)")

if writer: writer.close()
print(f"\n✅ DONE. Professional Dataset ready at {OUT_MERGED}")