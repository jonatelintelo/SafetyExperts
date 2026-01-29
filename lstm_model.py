import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class MoETraceClassifier(nn.Module):
    def __init__(self, num_total_experts, num_layers, top_k, embed_dim=16, hidden_dim=64):
        """
        Args:
            num_total_experts (int): Total unique experts (e.g., 8, 16, 64)
            num_layers (int): Depth of model (detected automatically)
            top_k (int): Experts per layer (detected automatically)
        """
        super().__init__()

        self.expert_embedding = nn.Embedding(num_embeddings=num_total_experts, embedding_dim=embed_dim)

        # Dynamic Input Size Calculation
        self.lstm_input_size = num_layers * top_k * embed_dim

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x shape: (Batch, Max_Tokens, Num_Layers, Top_K)
        batch_size, max_seq_len, _, _ = x.shape

        # Flatten: (Batch, Tokens, Layers*TopK*Embed)
        x_emb = self.expert_embedding(x)
        x_flat = x_emb.view(batch_size, max_seq_len, -1)

        packed_input = pack_padded_sequence(x_flat, lengths, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed_input)

        return self.classifier(ht[-1])
