from flask import Flask, render_template, request, jsonify
import torch
import faiss
import pickle
import numpy as np
from step3_model import SequentialTwoTower # Your architecture

app = Flask(__name__)

# --- Load Resources ---
with open("data/item_mapper.pkl", "rb") as f:
    item_list = pickle.load(f)
with open("data/title_to_idx.pkl", "rb") as f:
    title_to_idx = pickle.load(f)

idx_to_title = {v: k for k, v in title_to_idx.items()}
all_titles = list(title_to_idx.keys())

# Load Model
model = SequentialTwoTower(num_items=len(item_list), embedding_dim=64)
model.load_state_dict(torch.load("models/two_tower_epoch_3.pth", map_location="cpu"))
model.eval()

# Load FAISS Index
index = faiss.read_index("models/item_index.faiss")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/keyword_search')
def keyword_search():
    """Initial Search to find a Camera/Phone/etc."""
    query = request.args.get('q', '').lower()
    if len(query) < 3: return jsonify([])
    # Filter through 1M+ items
    matches = [{"title": t, "asin": item_list[title_to_idx[t]]} 
               for t in all_titles if query in t.lower()][:10]
    return jsonify(matches)

@app.route('/predict', methods=['POST'])
def predict():
    """Sequential Neural Recommendation based on 'Liked' items"""
    data = request.json
    liked_titles = data.get('titles', [])
    if not liked_titles: return jsonify([])

    indices = [title_to_idx[t] for t in liked_titles if t in title_to_idx]
    input_seq = torch.tensor([indices], dtype=torch.long)
    
    with torch.no_grad():
        # GRU generates Intent Vector
        user_vector = model(input_seq).numpy().astype('float32')
    
    faiss.normalize_L2(user_vector)
    _, results = index.search(user_vector, 6)
    
    recs = [{"title": idx_to_title.get(r, "Item"), "asin": item_list[r]} 
            for r in results[0][1:]]
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)