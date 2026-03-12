import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model_arch import SequentialTwoTower
import os

# 1. Configuration - Adjusted for CPU stability
EMBEDDING_DIM = 64
BATCH_SIZE = 128   # Reduced from 1024 so CPU doesn't hang
EPOCHS = 3         # 3 epochs is standard for convergence on smaller subsets
DEVICE = torch.device("cpu") # Explicitly using CPU

os.makedirs("models", exist_ok=True)

# 2. Load Data and take a strategic subset
print(f"📂 Loading sequences...")
full_df = pd.read_parquet("data/user_sequences.parquet")

# Strategic Subset: Taking 200,000 sequences is enough for a strong demo
# without crashing your computer.
df = full_df.head(200000) 

X = torch.tensor(df.iloc[:, :5].values, dtype=torch.long)
y = torch.tensor(df.iloc[:, 5].values, dtype=torch.long)
dataset = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

with open("data/item_mapper.pkl", "rb") as f:
    num_items = len(pickle.load(f))

# 3. Model Initialization
model = SequentialTwoTower(num_items, EMBEDDING_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Accuracy Metric Function
def get_hit_rate(logits, targets, k=10):
    _, top_indices = torch.topk(logits, k, dim=1)
    hits = (top_indices == targets.view(-1, 1)).any(dim=1).sum().item()
    return hits / targets.size(0)

# 5. The Training Loop
print(f"🚀 Starting Proper Training on {len(df):,} samples...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    total_hit_rate = 0
    
    for batch_idx, (seq, target) in enumerate(dataset):
        # Forward pass
        user_vector = model(seq)
        
        # In-Batch Negatives (The "Proper" way)
        # We compare the user vector against all 1.07M item embeddings
        logits = torch.matmul(user_vector, model.item_embeddings.weight.t())
        
        loss = criterion(logits, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track Hit Rate every 50 batches
        if batch_idx % 50 == 0:
            hit_rate = get_hit_rate(logits, target)
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataset)} | Loss: {loss.item():.4f} | Hit@10: {hit_rate*100:.2f}%")

    avg_loss = total_loss / len(dataset)
    print(f"✅ Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"models/two_tower_epoch_{epoch+1}.pth")

print("💾 Final Model saved successfully.")