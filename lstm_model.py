import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class HierarchicalMoEClassifier(nn.Module):
    def __init__(self, num_experts, embed_dim=16, path_hidden=32, prompt_hidden=64):
        super().__init__()
        
        # 1. Expert Embedding
        self.embedding = nn.Embedding(num_experts, embed_dim)
        
        # 2. INNER LSTM: Processes the path through 32 layers
        self.layer_lstm = nn.LSTM(input_size=embed_dim, 
                                  hidden_size=path_hidden, 
                                  batch_first=True)
        
        # 2. INNER LSTM alternative: Processes the path through 32 layers
        # Input: 2 experts per layer * embed_dim
        # self.layer_lstm = nn.LSTM(input_size=2 * embed_dim, 
        #                           hidden_size=path_hidden, 
        #                           batch_first=True)

        
        # 3. OUTER LSTM: Processes the sequence of tokens
        # Input: The summary vector from the layer_lstm
        self.prompt_lstm = nn.LSTM(input_size=path_hidden, 
                                   hidden_size=prompt_hidden, 
                                   batch_first=True)
        
        self.classifier = nn.Linear(prompt_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x shape: (Batch, Tokens, 32_Layers, 2_Experts)
        batch, tokens, layers, _ = x.shape
        
        # --- Step 1: Embed Experts ---
        # (Batch, Tokens, 32, 2, Embed)
        x_emb = self.embedding(x)

        # --- Sum across the 'k' dimension (dim=3) ---
        # This makes [1,3] identical to [3,1]
        # Shape becomes: (Batch, Tokens, 32, Embed_Dim)
        x_layer_input = x_emb.sum(dim=3)
        # Potentially the order agnosticism could also be achieved by soritng the experts in 'x' before the embedding.
        
        # # Flatten the 2 experts (combine them into one step input)
        # # (Batch, Tokens, 32, 2*Embed)
        # x_layer_input = x_emb.view(batch, tokens, layers, -1)
        
        # --- Step 2: Inner LSTM (Layer Path) ---
        # We need to fold Batch and Tokens together to process all paths at once
        # (Batch*Tokens, 32, Input_Dim)
        x_folded = x_layer_input.view(batch * tokens, layers, -1)
        
        # Run LSTM over the 32 layers
        _, (ht_path, _) = self.layer_lstm(x_folded)
        
        # ht_path is (1, Batch*Tokens, Path_Hidden)
        # Unfold back to (Batch, Tokens, Path_Hidden)
        token_path_embeddings = ht_path[-1].view(batch, tokens, -1)
        
        # --- Step 3: Outer LSTM (Token Sequence) ---
        # Now we proceed exactly like the previous model
        packed_input = pack_padded_sequence(token_path_embeddings, lengths, 
                                            batch_first=True, enforce_sorted=False)
        
        _, (ht_prompt, _) = self.prompt_lstm(packed_input)
        
        logits = self.classifier(ht_prompt[-1])
        
        return self.sigmoid(logits)
    

class MoETraceClassifier(nn.Module):
    def __init__(self, num_total_experts, num_layers, top_k, embed_dim=16, hidden_dim=64):
        """
        Args:
            num_total_experts (int): Total unique experts (e.g., 8, 16, 64)
            num_layers (int): Depth of model (detected automatically)
            top_k (int): Experts per layer (detected automatically)
        """
        super().__init__()
        
        self.expert_embedding = nn.Embedding(num_embeddings=num_total_experts, 
                                             embedding_dim=embed_dim)

        # Dynamic Input Size Calculation
        self.lstm_input_size = num_layers * top_k * embed_dim
        
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
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