import torch
import torch.nn as nn

class SequentialTwoTower(nn.Module):
    def __init__(self, num_items, embedding_dim=64):
        super(SequentialTwoTower, self).__init__()
        
        # 1. Item Tower: Represents 1.07M products as vectors in 3D space
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # 2. User Tower: GRU acts as "Short-Term Memory" to understand the journey
        self.user_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        
        # Final layer to produce the "Intent Vector"
        self.user_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sequence):
        # sequence shape: (batch_size, 5)
        item_embs = self.item_embeddings(sequence)
        
        # GRU understands the order of clicks (e.g., Laptop then Mouse)
        _, hidden = self.user_gru(item_embs)
        
        # Output is the vector representing the user's current mood/intent
        user_vector = self.user_fc(hidden.squeeze(0))
        return user_vector