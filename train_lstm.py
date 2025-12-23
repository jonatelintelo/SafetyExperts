import model_configurations
import data_utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import lstm_model
import lstm_data
from torch.utils.data import random_split
import find_safety_experts
import model_utils


SEED = 42

# Torch CPU & CUDA
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

# validation_split=0.2 means 20% of data is used for validation
def train(traces, labels, num_total_experts):
    # --- Config ---
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 2
    
    # 1. Initialize the FULL Dataset first
    # This runs your auto-detection logic once on all data
    full_dataset = lstm_data.MoETraceDataset(traces, labels)
    
    # 2. Extract detected params NOW (Before splitting)
    # We access these from the full_dataset directly
    DETECTED_LAYERS = full_dataset.detected_num_layers
    DETECTED_TOP_K = full_dataset.detected_top_k
    
    # 3. Perform the Split
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    
    # random_split wraps your dataset into two "Subset" objects
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 4. Create DataLoaders from the splits
    train_loader = lstm_data.get_dataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = lstm_data.get_dataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model
    model = lstm_model.MoETraceClassifier(
        num_total_experts=num_total_experts,
        num_layers=DETECTED_LAYERS,
        top_k=DETECTED_TOP_K
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"🚀 Starting training for {DETECTED_LAYERS}-layer model with Top-{DETECTED_TOP_K} routing.")
    print(f"📊 Data split: {len(train_dataset)} Train | {len(val_dataset)} Validation\n")
    
    # --- Outer Loop (Epochs) ---
    for epoch in range(EPOCHS):
        
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train()  # IMPORTANT: Enable dropout/batchnorm
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Using enumerate to fix the "current_loss" calculation
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for i, (x_batch, y_batch, lengths) in pbar:
            optimizer.zero_grad()
            
            # Forward & Backward
            logits = model(x_batch, lengths).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)
            
            # Dynamic Progress Bar Update
            # Note: Divide by (i+1) for accurate running average, not len(dataloader)
            current_train_loss = running_loss / (i + 1)
            current_train_acc = correct_preds / total_samples
            
            pbar.set_postfix({'loss': f'{current_train_loss:.4f}', 'acc': f'{current_train_acc:.2%}'})
        
        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval()  # IMPORTANT: Disable dropout/batchnorm
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Disable gradient calculation for speed and memory efficiency
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                logits = model(x_batch, lengths).squeeze()
                loss = criterion(logits, y_batch)
                
                val_running_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        # Calculate final validation metrics for the epoch
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        
        # Print summary using tqdm.write to avoid breaking the progress bar
        tqdm.write(f"   🟢 Validation: Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.2%}")
        tqdm.write("-" * 50)

    print("\n✅ Training complete.")
    return model

if __name__ == "__main__":
    model_id = 1

    models = [
        # LLMs
        "Qwen/Qwen3-30B-A3B-Instruct-2507",     #0
        "microsoft/Phi-3.5-MoE-instruct",       #1
        "mistralai/Mixtral-8x7B-Instruct-v0.1", #2
        "openai/gpt-oss-20b",                   #3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",          #4
        "tencent/Hunyuan-A13B-Instruct",        #5
        "deepseek-ai/deepseek-moe-16b-chat",    #6
        "IntervitensInc/pangu-pro-moe-model",   #7
        # VLMs
        "moonshotai/Kimi-VL-A3B-Instruct",      #8
        "moonshotai/Kimi-VL-A3B-Thinking",      #9
        "moonshotai/Kimi-VL-A3B-Thinking-2506", #10
        "deepseek-ai/deepseek-vl2-small",       #11
        "deepseek-ai/deepseek-vl2",             #12
    ]
    
    model_config = model_configurations.models[models[model_id]]

    print(f"Tested Model: '{model_config.model_name}'")
    
    model, tokenizer = model_utils.load_model(models[model_id])
    
    print("\nLoading precomputed data...")
    traces = data_utils.load_data(BASE_FOLDER + f"{model_config.model_name}/data/{model_config.model_name}_input.pkl")
    labels = data_utils.load_data(BASE_FOLDER + f"{model_config.model_name}/data/{model_config.model_name}_labels.pkl")

    trained_lstm_model = train(traces, labels, num_total_experts=model_config.num_router_expert)

    questions, labels = data_utils.load_twinset()
    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

##############################################################################
# Print some example trigger token analyses for malicious and benign prompts #
##############################################################################

    # # Collect some examples of pairs in the twin set
    # for prompt_index in [0,1,2,3,4,250,251,252,253,254]:
    #     print(f"questions[prompt_index]: {questions[prompt_index]}")
    #     risk_scores = find_safety_experts.analyze_risk_trajectory(trained_lstm_model, traces, prompt_idx=prompt_index)
    #     find_safety_experts.trigger_token_analysis(tokenizer, questions[prompt_index], risk_scores)

#################################################################################################
# Prune the experts that process the token with highest probability in maliciousness trajectory #
# Pruning takes place on all layers.                                                            #
#################################################################################################

    # for prompt_index in range(50):
    #     print(f"Question: {questions[prompt_index]}")
    #     responses = model_utils.generate_output(model, model_config.model_name, tokenizer, [prompts[prompt_index]], max_new_tokens=256, batch_size=128)
    #     print(f"Response before pruning: {responses}")
    #     risk_scores = find_safety_experts.analyze_risk_trajectory(trained_lstm_model, traces, prompt_idx=prompt_index)
    #     result = find_safety_experts.print_risk_trajectory(tokenizer, questions[prompt_index], risk_scores)
    #     refusal_experts = traces[prompt_index][result]
    #     print(f"Experts to prune: {refusal_experts}")
    #     # Example: Experts to prune: [[7, 15], [12, 8], [3, 6], [7, 1], [1, 9], [10, 14], [11, 3], [15, 5], [7, 5], [12, 5], [3, 12], [5, 7], [9, 8], [5, 0], [12, 11], [2, 14], [7, 11], [3, 15], [13, 10], [12, 7], [10, 3], [9, 1], [9, 6], [0, 14], [15, 2], [11, 7], [13, 10], [9, 13], [2, 14], [8, 9], [0, 5], [1, 5]]
    #     # Example shape: list of (num_layers, Top-K)
    #     pruning_handles = model_utils.register_pruning_hooks_token(model, refusal_experts, model_config.gate_name)
    #     responses = model_utils.generate_output(model, model_config.model_name, tokenizer, [prompts[prompt_index]], max_new_tokens=256, batch_size=128)
    #     print(f"Response after pruning: {responses}\n")

    #     for handle in pruning_handles:
    #             handle.remove()

##############################################################################################
# Pruning layer-wise top experts with gradients.
# e.g.
##############################################################################################

    # full_dataset = lstm_data.MoETraceDataset(traces, labels)
    # full_loader = lstm_data.get_dataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # for prompt_index, (x_batch, y_batch, lengths) in enumerate(full_loader):
    #     responses = model_utils.generate_output(model, model_config.model_name, tokenizer, [prompts[prompt_index]], max_new_tokens=256, batch_size=128)
    #     print(f"Question: {questions[prompt_index]}")
    #     print(f"Response before pruning: {responses}")

    #     importance_map, expert_ids = find_safety_experts.get_expert_importance(trained_lstm_model, x_batch, lengths)
    #     rankings = find_safety_experts.find_top_refusal_experts(importance_map, expert_ids)

    #     print(f"Rankings of important experts for prompt {prompt_index}: {rankings}")
    #     # Example: Rankings of important experts for prompt 0: [((20, 3), 0.24483465735102072), ((24, 15), 0.2028980702161789), ((22, 6), 0.13903061170276487), ((21, 9), 0.13783680980850477), ((23, 14), 0.13004399169949465), ((31, 5), 0.1261957463393628), ((18, 10), 0.11587553062054212), ((13, 0), 0.1149328202009201), ... ]
    #     # Example shape: 1-D list of (num_layers*num_experts)
    #     # Each item is ((layer_idx,expert_idx),score)

    #     # Loop determines how many experts in layers we prune, e.g., i = 10 means we prune the first 10 occurences.
    #     for i in range(1, len(rankings)+1):
    #         candidates = [item[0] for item in rankings[:i]] 
    #         candidates = sorted(candidates, key=lambda x: x[0])
    #         candidates = [(f"{str(item[0])}."+model_config.gate_name, item[1]) for item in candidates]
    #         # Candidates now contains (layer_suffix,expert_idx)
            
    #         pruning_handles = model_utils.register_pruning_hooks_candidates(model_config.model_name, model, candidates)

    #         responses = model_utils.generate_output(model, model_config.model_name, tokenizer, [prompts[prompt_index]], max_new_tokens=256, batch_size=128)
    #         print(f"Response after pruning index '{i}': {responses}")

    #         for handle in pruning_handles:
    #             handle.remove()

    #     # Break loop early for testing purposes
    #     if prompt_index == 10:
    #         break

#################################################################################################
#################################################################################################

    # # 3. Perform the Split
    # total_size = len(full_dataset)
    # val_size = int(total_size * 0.2)
    # train_size = total_size - val_size
    
    # # random_split wraps your dataset into two "Subset" objects
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 4. Create DataLoaders from the splits
    # train_loader = lstm_data.get_dataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = lstm_data.get_dataLoader(val_dataset, batch_size=1, shuffle=False)

    # for x_batch, y_batch, lengths in val_loader:
    #     importance_map, expert_ids = find_safety_experts.get_expert_importance(model, x_batch, lengths)
    #     results = find_safety_experts.find_top_expert_per_layer(importance_map, expert_ids)

    #     break

    # print(f"{'Layer':<10} | {'Top Expert':<12} | {'Score':<10}")
    # print("-" * 40)
    # for res in results:
    #     print(f"{res['layer']:<10} | {res['expert_id']:<12} | {res['importance_score']:.4f}")

    # questions, labels = data_utils.load_datasets(malicious_only=False) # label: 1 = unsafe

############################################################
############################################################

    # full_loader = lstm_data.get_dataLoader(full_dataset, batch_size=1, shuffle=False)

    # for prompt_index, (x_batch, y_batch, lengths) in enumerate(full_loader):
        # importance_map, expert_ids = find_safety_experts.get_expert_importance(trained_lstm_model, x_batch, lengths)

        # for i in range(16):
        #     rankings = find_safety_experts.find_global_top_expert(importance_map, expert_ids)
        #     refusal_drivers = sorted(
        #         [item for item in rankings if item[1] > 0], 
        #         key=lambda x: x[1], 
        #         reverse=True
        #     )
        #     refusal_drivers = [item[0] for item in refusal_drivers][:i]
        #     print(refusal_drivers)
        #     pruning_handles = model_utils.register_pruning_hooks(model, refusal_drivers, model_config.gate_name)
            
        #     responses = model_utils.generate_output(model, model_config.model_name, tokenizer, [prompts[prompt_index]], max_new_tokens=256, batch_size=128)
        #     print(f"Response after pruning index '{i}': {responses}")

        #     for handle in pruning_handles:
        #         handle.remove()

############################################################
############################################################



    # print("Top Experts Contributing to Refusal:")
    # for (layer, eid), score in rankings[:5]:
    #     print(f"Layer {layer} | Expert {eid} | Importance: {score:.4f}")

    print("\n------------------ Job Finished ------------------")
