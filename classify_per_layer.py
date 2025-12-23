import pandas as pd
import torch
from collections import defaultdict

def profile_expert_risk_layered(labels, traces, num_layers, num_experts_per_layer):
    """
    Calculates a 'Maliciousness Score' for every expert, distinguishing
    them by their layer, based on the Tracer LSTM's predictions.

    Args:
        model (torch.nn.Module): The trained Token Tracer LSTM model.
        traces (dict): Trace data where values are structured as 
                       [Seq_Len, Layers, Top_K].
        num_layers (int): The number of MoE layers in the model.
        num_experts_per_layer (int): The number of experts in each MoE layer.

    Returns:
        pd.DataFrame: A DataFrame with layer-specific Risk_Scores.
    """

    # Store accumulators using a dictionary of dictionaries for (Layer_ID, Expert_ID)
    # {layer_id: {expert_id: [risk_sum, count]}}
    expert_risk_map = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    with torch.no_grad():
        for pid, token_dict in traces.items():
            # 1. Prepare input
            token_indices = sorted(token_dict.keys())
            prompt_data = [token_dict[t] for t in token_indices]

            # Convert to tensor (1, Seq_Len, Layers, Top_K)
            # The innermost data structure should be [Layers, Top_K]
            x_tensor = torch.tensor(prompt_data, dtype=torch.long).unsqueeze(0)
            # length = [len(prompt_data)]
            
            # Check shape conformance
            if x_tensor.shape[2] != num_layers:
                print(f"Warning: Trace shape Layer dim ({x_tensor.shape[2]}) doesn't match NUM_LAYERS ({num_layers}). Skipping prompt {pid}.")
                continue
            
            # # 2. Get Model Prediction (Probability of Maliciousness)
            # # The LSTM processes the entire prompt trace and outputs a single maliciousness score (logit)
            # logits = model(x_tensor, length)
            # # Apply sigmoid to get probability P(Malicious | Trace)
            # prob = torch.sigmoid(logits).item()
            prob = labels[pid]

            # 3. Attribute this probability to every expert present in the trace, PER LAYER
            
            # Iterate through each MoE layer
            for layer_idx in range(num_layers):
                # Slice the tensor to get the experts activated in this layer across all tokens
                # Shape: (Seq_Len, Top_K)
                layer_experts = x_tensor[0, :, layer_idx, :]
                
                # Get the unique expert IDs used within this specific layer for this prompt
                unique_experts_in_layer = torch.unique(layer_experts)

                for exp_id in unique_experts_in_layer:
                    exp_id = exp_id.item()
                    
                    # Ensure the expert ID is valid (i.e., not padding or OOB)
                    if 0 <= exp_id < num_experts_per_layer:
                        expert_risk_map[layer_idx][exp_id][0] += prob  # Accumulate Risk Sum
                        expert_risk_map[layer_idx][exp_id][1] += 1      # Accumulate Count

    # 4. Calculate Average Risk Score and build the final DataFrame
    data = []
    for layer_id, experts in expert_risk_map.items():
        for expert_id, (risk_sum, count) in experts.items():
            if count > 0:
                avg_score = risk_sum / count
                data.append({
                    'Layer_ID': layer_id,
                    'Expert_ID': expert_id,
                    'Risk_Score': avg_score,
                    'Frequency': count
                })

    if not data:
        return pd.DataFrame(columns=['Layer_ID', 'Expert_ID', 'Risk_Score', 'Frequency'])
        
    # 5. Create and Sort the DataFrame
    df = pd.DataFrame(data)
    
    # Sort: Top of list = Most Malicious, Bottom = Safest. Sort primarily by score, 
    # then by frequency for tie-breaking.
    df = df.sort_values(by=['Risk_Score', 'Frequency'], ascending=[False, False])
    
    return df