import torch

def print_risk_trajectory(tokenizer, prompt, probs_sequence):
    """
    Aligns tokens with probability scores and prints them as (Score, Token) tuples.
    """
    # 1. Tokenize (Ensure special tokens match your MoE trace collection)
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    for i, (_, prob) in enumerate(zip(tokens, probs_sequence)):
        current_highest_prob = 0
        if prob > current_highest_prob:
            result = i
    return result
        
def trigger_token_analysis(tokenizer, prompt, probs_sequence):
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    for i, (token, prob) in enumerate(zip(tokens, probs_sequence)):
        print(f"{i:<6} | ({prob:.4f}, '{token}')")

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


def find_global_top_expert(importance_map, expert_ids):
    """
    importance_map: (Tokens, Layers, Top_K) - Scalar gradients
    expert_ids: (Tokens, Layers, Top_K) - Original expert indices
    """
    tokens, layers, k_val = importance_map.shape
    global_expert_scores = {}

    for t in range(tokens):
        for l in range(layers):
            for k in range(k_val):
                eid = expert_ids[t, l, k].item()
                score = importance_map[t, l, k].item()

                if eid not in global_expert_scores:
                    global_expert_scores[eid] = 0.0
                global_expert_scores[eid] += score

    # Convert to a list of dictionaries and sort by score descending
    ranked_experts = [
        (eid, score) for eid, score in global_expert_scores.items()
    ]
    
    return ranked_experts

def find_top_refusal_experts(importance_map, expert_ids):
    # expert_ids and importance_map are both (Tokens, Layers, Top_K)
    expert_culprits = {}

    tokens, layers, k_val = importance_map.shape

    for l in range(layers):
        for t in range(tokens):
            for k in range(k_val):
                eid = expert_ids[t, l, k].item()
                score = importance_map[t, l, k].item()
                
                # Unique key is (Layer, ExpertID)
                key = (l, eid)
                if key not in expert_culprits:
                    expert_culprits[key] = 0.0
                expert_culprits[key] += score

    # Sort by highest positive score (most responsible for Label 1)
    sorted_experts = sorted(expert_culprits.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_experts
    
def find_top_expert_per_layer(importance_map, expert_ids):
    """
    importance_map: (Tokens, Layers, Top_K) - Scalar gradients
    expert_ids: (Tokens, Layers, Top_K) - Original expert indices
    """
    tokens, layers, k_val = importance_map.shape
    top_experts_by_layer = []

    for l in range(layers):
        # Dictionary to store summed scores for experts specific to THIS layer
        layer_expert_scores = {}

        for t in range(tokens):
            for k in range(k_val):
                eid = expert_ids[t, l, k].item()
                score = importance_map[t, l, k].item()

                if eid not in layer_expert_scores:
                    layer_expert_scores[eid] = 0.0
                layer_expert_scores[eid] += score

        # Find the expert with the max score in this layer
        if layer_expert_scores:
            top_eid = max(layer_expert_scores, key=layer_expert_scores.get)
            top_score = layer_expert_scores[top_eid]
            top_experts_by_layer.append({
                "layer": l,
                "expert_id": top_eid,
                "importance_score": top_score
            })

    return top_experts_by_layer

def get_expert_importance(model, x, lengths):
    """
    input:
    model: LSTM model
    x: (1, Max_Tokens, Num_Layers, Top_K) - Integer Expert IDs
    lengths: (1,) tensor - Number of tokens

    output:
    importance_map: (Tokens, Layers, Top_K) - Scalar gradients
    expert_ids: (Tokens, Layers, Top_K) - Original expert indices
    """
    model.eval()
    model.zero_grad()

    # 1. Manually embed the input
    # x_emb shape: (1, Tokens, Layers, Top_K, Embed_Dim)
    x_emb = model.expert_embedding(x)
    x_emb = x_emb.detach().requires_grad_(True)
    
    # 2. Replicate the forward pass logic
    batch_size, max_seq_len, num_layers, top_k, embed_dim = x_emb.shape
    x_flat = x_emb.view(batch_size, max_seq_len, -1)
    
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(
        x_flat, lengths, batch_first=True, enforce_sorted=False
    )
    _, (ht, _) = model.lstm(packed_input)
    logit = model.classifier(ht[-1]) # This is the "Refusal Score"

    # 3. Backward pass to get gradients w.r.t embeddings
    logit.backward()
    
    # 4. Calculate Gradient * Input (Importance Score)
    # Shape: (1, Tokens, Layers, Top_K, Embed_Dim)
    importance_tensor = x_emb.grad * x_emb
    
    # 5. Aggregate across the embedding dimension to get one score per expert
    # Shape: (Tokens, Layers, Top_K)
    scalar_importance = importance_tensor.sum(dim=-1).squeeze(0)
    
    return scalar_importance, x.squeeze(0)