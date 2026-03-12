import torch
import faiss
import pickle
import numpy as np
from model_arch import SequentialTwoTower # Import the architecture

# 1. Load the Item Mapper to get the correct item count
print("📂 Loading item mapper...")
with open("data/item_mapper.pkl", "rb") as f:
    unique_asins = pickle.load(f)
num_items = len(unique_asins)

# 2. Load the Trained Model
print("🧠 Loading trained model weights...")
model = SequentialTwoTower(num_items=num_items, embedding_dim=64)
model.load_state_dict(torch.load("models/two_tower_epoch_3.pth", map_location="cpu"))
model.eval()

# 3. Extract Item Embeddings (The "Store Map")
# We take the weights from the Item Tower
print("🚀 Extracting item vectors...")
with torch.no_grad():
    item_vectors = model.item_embeddings.weight.detach().cpu().numpy().astype('float32')

# 4. Normalize vectors for Cosine Similarity
faiss.normalize_L2(item_vectors)

# 5. Create and Build the FAISS Index
# Inner Product (IP) on normalized vectors = Cosine Similarity
index = faiss.IndexFlatIP(64) 
index.add(item_vectors)

# 6. Save the Index
faiss.write_index(index, "models/item_index.faiss")
print(f"✅ Success! FAISS index created with {index.ntotal:,} items.")
print("💾 Saved to: models/item_index.faiss")