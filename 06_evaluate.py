import torch
import numpy as np
import faiss
import pickle
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model_arch import SequentialTwoTower # Your architecture

# --- 1. Metric Calculation Functions ---
def get_metrics(top_indices, targets, k=10):
    hits = 0
    rr_sum = 0
    for i in range(len(targets)):
        if targets[i] in top_indices[i]:
            hits += 1
            # MRR Calculation: 1 / Rank (1-based)
            rank = np.where(top_indices[i] == targets[i])[0][0] + 1
            rr_sum += (1.0 / rank)
    return hits, rr_sum

# --- 2. Main Evaluation Logic ---
def run_full_evaluation(model, test_loader, index, num_total_items, k=10):
    model.eval()
    total_hits, total_mrr, total_count = 0, 0, 0
    recommended_items = set() # To track Catalog Coverage
    user_results = []         # To track Personalization

    print(f"🔍 Scanning 1.07M products for {len(test_loader.dataset)} test samples...")
    
    with torch.no_grad():
        for seqs, targets in tqdm(test_loader):
            # Step A: Generate User Embedding via GRU
            user_vectors = model(seqs).cpu().numpy().astype('float32')
            faiss.normalize_L2(user_vectors)
            
            # Step B: Vector Search across the 1.07M item catalog
            _, top_k_results = index.search(user_vectors, k)
            
            # Step C: Update Accuracy Metrics (Hit Rate & MRR)
            hits, mrr_val = get_metrics(top_k_results, targets.numpy(), k)
            total_hits += hits
            total_mrr += mrr_val
            total_count += len(targets)
            
            # Step D: Track uniqueness for Coverage and Personalization
            for result in top_k_results:
                recommended_items.update(result)
                user_results.append(tuple(result))

    # --- 3. Calculate Final Scores ---
    hit_rate = (total_hits / total_count) * 100
    mrr = (total_mrr / total_count) * 100
    
    # Catalog Coverage: % of items in your database that the AI actually uses
    coverage = (len(recommended_items) / num_total_items) * 100
    
    # Personalization: % of users who get unique lists
    unique_lists = len(set(user_results))
    personalization = (unique_lists / total_count) * 100

    print(f"\n" + "="*40)
    print(f"✨ MAJOR PROJECT EVALUATION REPORT ✨")
    print(f"="*40)
    print(f"📈 Hit Rate@10:     {hit_rate:.4f}%  (Accuracy)")
    print(f"📉 MRR@10:          {mrr:.4f}%  (Ranking Quality)")
    print(f"🌐 Catalog Coverage: {coverage:.4f}%  (Model Diversity)")
    print(f"👤 Personalization:  {personalization:.4f}%  (User Uniqueness)")
    print(f"="*40)

# --- 4. Load Resources & Execute ---
if __name__ == "__main__":
    print("📂 Loading trained model, FAISS index, and item maps...")
    
    # Load Item Mapper
    with open("data/item_mapper.pkl", "rb") as f:
        item_list = pickle.load(f)
        num_items = len(item_list)

    # Load Trained Model
    model = SequentialTwoTower(num_items=num_items, embedding_dim=64)
    model.load_state_dict(torch.load("models/two_tower_epoch_3.pth", map_location="cpu"))
    
    # Load FAISS Index
    index = faiss.read_index("models/item_index.faiss")

    # Load Test Data (last 5,000 sequences)
    full_df = pd.read_parquet("data/user_sequences.parquet")
    test_df = full_df.tail(5000) 
    
    X_test = torch.tensor(test_df.iloc[:, :5].values, dtype=torch.long)
    y_test = torch.tensor(test_df.iloc[:, 5].values, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    run_full_evaluation(model, test_loader, index, num_items)