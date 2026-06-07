import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from collections import defaultdict

# Import modules from our codebase
import argument_parser
import moe_models.model_configurations as model_configurations
import data.data_utils as data_utils
import lstm.lstm_model as lstm_model
import lstm.lstm_data as lstm_data

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
    model.train()
    model.zero_grad()

    # Manually embed the input
    # x_emb shape: (1, Tokens, Layers, Top_K, Embed_Dim)
    x_emb = model.expert_embedding(x)
    x_emb = x_emb.detach().requires_grad_(True)

    # Replicate the forward pass logic
    batch_size, max_seq_len, num_layers, top_k, embed_dim = x_emb.shape
    x_flat = x_emb.view(batch_size, max_seq_len, -1)

    packed_input = torch.nn.utils.rnn.pack_padded_sequence(x_flat, lengths, batch_first=True, enforce_sorted=False)
    _, (ht, _) = model.lstm(packed_input)
    logit = model.classifier(ht[-1])  # This is the "Refusal Score"

    # Backward pass to get gradients w.r.t embeddings
    logit.backward()

    # Calculate Gradient * Input (Importance Score)
    # Shape: (1, Tokens, Layers, Top_K, Embed_Dim)
    importance_tensor = x_emb.grad * x_emb

    # Aggregate across the embedding dimension to get one score per expert
    # Shape: (Tokens, Layers, Top_K)
    scalar_importance = importance_tensor.sum(dim=-1).squeeze(0)

    return scalar_importance, x.squeeze(0)

def accumulate_expert_importance(importance_map, expert_ids, global_culprits):
    """
    Vectorized version of find_top_refusal_experts.
    Directly adds to the global dictionary to avoid redundant sorting per prompt.
    """
    tokens, layers, k_val = importance_map.shape

    # Create a tensor of layer indices that matches the shape of our data
    # Shape: (Tokens, Layers, Top_K)
    layer_indices = torch.arange(layers, device=expert_ids.device).view(1, layers, 1).expand(tokens, layers, k_val)

    # Flatten everything to 1D lists for lightning-fast zipping
    flat_layers = layer_indices.flatten().tolist()
    flat_eids = expert_ids.flatten().tolist()
    flat_scores = importance_map.flatten().tolist()

    # Accumulate globally
    for l, eid, score in zip(flat_layers, flat_eids, flat_scores):
        global_culprits[(l, eid)] += score


if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
        "IntervitensInc/pangu-pro-moe-model",  # 7
        "allenai/OLMoE-1B-7B-0125-Instruct",  # 8
    ]

    model_config = model_configurations.models[models[model_id]]
    print(f"\n🔍 Analyzing Refusal Experts for: {model_config.model_name}")

    # Load the Traces & Labels
    lstm_input_dir = os.path.join(root_folder, "data", "lstm_input", model_config.model_name)
    traces = data_utils.load_data(os.path.join(lstm_input_dir, f"{model_config.model_name}_traces.pkl"))
    labels = data_utils.load_data(os.path.join(lstm_input_dir, f"{model_config.model_name}_labels.pkl"))

    dataset = lstm_data.MoETraceDataset(traces, labels)
    
    # Batch size MUST be 1 for the get_expert_importance backward pass to work per-sequence
    data_loader = lstm_data.get_dataLoader(dataset, batch_size=1, shuffle=False)

    # Load the Trained Model
    lstm_model_path = os.path.join(root_folder, "lstm", "trained_lstm_models", model_config.model_name, f"{model_config.model_name}_lstm.pkl")

    checkpoint = torch.load(lstm_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = lstm_model.MoETraceClassifier(
        num_total_experts=checkpoint["num_total_experts"],
        num_layers=checkpoint["num_layers"],
        top_k=checkpoint["top_k"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Global Dictionary to store cumulative scores
    global_expert_culprits = defaultdict(float)

    print("\n⚙️ Extracting gradient attributions...")
    
    # Process every trace
    for x_batch, y_batch, lengths in tqdm(data_loader, total=len(data_loader)):
        
        # Optionally: If you ONLY want to measure the gradients of malicious prompts, 
        # you can uncomment the following lines:
        if y_batch.item() == 0.0:
            continue
            
        x_batch = x_batch.to(device)
        # lengths stays on CPU
        
        # Extract map
        importance_map, expert_ids = get_expert_importance(model, x_batch, lengths)
        
        # Accumulate
        accumulate_expert_importance(importance_map, expert_ids, global_expert_culprits)

    # Sort the globally aggregated scores
    # Higher positive score = heavier push toward Refusal (Label 1)
    sorted_refusal_experts = sorted(global_expert_culprits.items(), key=lambda x: x[1], reverse=True)

    if print_logging:
        print("\n🏆 Top 10 Most Responsible Refusal Experts (Layer, ExpertID): Score")
        for i in range(min(10, len(sorted_refusal_experts))):
            expert, score = sorted_refusal_experts[i]
            print(f"   Rank {i+1}: {expert} -> {score:.4f}")

    # Save the results
    output_dir = os.path.join(root_folder, "refusal_experts", model_config.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{model_config.model_name}_refusal_experts.pkl")
    data_utils.save_data(sorted_refusal_experts, save_path)
    
    print(f"\n💾 Saved refusal expert rankings to {save_path}")
    print("\n------------------ Job Finished ------------------")