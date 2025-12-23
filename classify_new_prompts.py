import numpy as np
import pandas as pd
import torch

############################################################################################
def explain_specific_prompt(model, traces, prompt_idx):
    """
    Identifies which experts in a specific prompt contributed most to the classification.
    """
    model.train() # Gradients need train mode (or grad enabled)
    
    # 1. Prepare Data
    token_indices = sorted(traces[prompt_idx].keys())
    prompt_data = [traces[prompt_idx][t] for t in token_indices]
    x_tensor = torch.tensor(prompt_data, dtype=torch.long).unsqueeze(0) # (1, Tokens, Layers, TopK)
    length = [len(prompt_data)]
    
    # 2. Hook into Embeddings
    # We need to access the embeddings to calculate gradients on them
    # Since we can't differentiate indices, we do it after the embedding lookup
    
    # Embed manually so we can track gradients on the embedding tensor
    emb_weight = model.expert_embedding.weight
    x_emb = torch.nn.functional.embedding(x_tensor, emb_weight)
    
    # Enable gradients on this specific input's embedding representation
    x_emb.retain_grad()
    
    # 3. Forward Pass (Manual parts of the LSTM forward)
    # Flatten
    b, tokens, layers, k, dim = x_emb.shape
    x_flat = x_emb.view(b, tokens, -1)
    
    # Run LSTM
    # Note: If you used pack_padded_sequence in forward, we replicate that logic,
    # or just run raw if batch=1 (simpler)
    lstm_out, (ht, _) = model.lstm(x_flat)
    final_state = ht[-1]
    
    logits = model.classifier(final_state)
    prob = torch.sigmoid(logits)
    
    # 4. Backward Pass (Saliency)
    # We want to know what maximizes the "Malicious" class (Logit)
    logits.backward()
    
    # 5. Analyze Gradients
    # Gradient shape: (1, Tokens, Layers, Top_K, Embed_Dim)
    grads = x_emb.grad
    
    # Aggregate gradients: sum absolute values across embedding dimension
    # This gives a single "Importance Score" for every expert token
    # Shape: (Tokens, Layers, Top_K)
    importance_scores = grads.abs().sum(dim=-1).squeeze(0)
    
    # 6. Find the single most influential Expert instance
    # Flatten to find max
    max_idx = torch.argmax(importance_scores)
    
    # Unravel index to get (Token, Layer, Expert_Index_0_or_1)
    # Note: logic depends on tensor flattening, simplistic approach below:
    
    flat_imp = importance_scores.flatten()
    top_indices = torch.topk(flat_imp, 5).indices
    
    print(f"--- Analysis for Prompt {prompt_idx} (Prob: {prob.item():.4f}) ---")
    
    for idx in top_indices:
        # Convert flat index back to dimensions
        # Shape was (Tokens, Layers, TopK)
        k_idx = idx.item() % 2
        rem = idx.item() // 2
        layer_idx = rem % 32 # Assuming 32 layers
        token_idx = rem // 32
        
        expert_id = x_tensor[0, token_idx, layer_idx, k_idx].item()
        score = flat_imp[idx].item()
        
        print(f"Token {token_idx} | Layer {layer_idx} | Expert {expert_id} | Importance: {score:.4f}")

############################################################################################

def analyze_risk_trajectory(model, trace_dict, prompt_idx):
    """
    Feeds a single prompt's traces into the model and plots the maliciousness 
    score at every token step.
    """
    model.eval() # Set to evaluation mode
    
    # 1. Prepare Single Input
    # Extract the list of layer-traces for this specific prompt
    token_indices = sorted(trace_dict[prompt_idx].keys())
    # print(f"token_indices: {token_indices}")
    single_prompt_data = [trace_dict[prompt_idx][t] for t in token_indices]
    # print(f"single_prompt_data: {single_prompt_data}")
    
    # Convert to Tensor & Add Batch Dimension -> (1, Num_Tokens, Num_Layers, Top_K)
    x_tensor = torch.tensor(single_prompt_data, dtype=torch.long).unsqueeze(0)
    
    # Get actual length
    length = [len(single_prompt_data)]

    with torch.no_grad():
        # --- REPLICATING FORWARD PASS MANUALLY TO GET INTERMEDIATE STEPS ---
        
        # 1. Embed & Flatten
        x_emb = model.expert_embedding(x_tensor) 
        # (1, Seq_Len, Input_Size)
        x_flat = x_emb.view(1, length[0], -1) 
        
        # 2. Run LSTM
        # We DO NOT use pack_padded_sequence here because batch_size is 1 (no padding needed)
        # output shape: (1, Seq_Len, Hidden_Dim)
        # This 'output' contains the hidden state for *every* token
        lstm_output, _ = model.lstm(x_flat)
        
        # 3. Classify EVERY step
        # We apply the Linear layer to the entire sequence
        # Shape: (1, Seq_Len, 1)
        logits_sequence = model.classifier(lstm_output)
        
        # 4. Convert to Probabilities
        probs_sequence = torch.sigmoid(logits_sequence).squeeze().tolist()
    
    return probs_sequence
