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
import numpy as np
from datasets import load_dataset
from argument_parser import parse_arguments
import sys
from collections import defaultdict

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train(traces, labels, num_total_experts):
    # --- Config ---
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 10

    # 1. Initialize the FULL Dataset first
    # This runs your auto-detection logic once on all data
    dataset = lstm_data.MoETraceDataset(traces, labels)

    # 2. Extract detected params NOW (Before splitting)
    # We access these from the dataset directly
    DETECTED_LAYERS = dataset.detected_num_layers
    DETECTED_TOP_K = dataset.detected_top_k

    # 3. Perform the Split
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    # random_split wraps your dataset into two "Subset" objects
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. Create DataLoaders from the splits
    train_loader = lstm_data.get_dataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = lstm_data.get_dataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model
    model = lstm_model.MoETraceClassifier(
        num_total_experts=num_total_experts,
        num_layers=DETECTED_LAYERS,
        top_k=DETECTED_TOP_K,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n🚀 Starting training for {DETECTED_LAYERS}-layer model with Top-{DETECTED_TOP_K} routing.")
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
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
        )

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

            pbar.set_postfix({"loss": f"{current_train_loss:.4f}", "acc": f"{current_train_acc:.2%}"})

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
    arguments = parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    models = [
        # LLMs
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
        "IntervitensInc/pangu-pro-moe-model",  # 7
    ]

    model_config = model_configurations.models[models[model_id]]

    print(f"\nTested Model: {model_config.model_name}")

    model, tokenizer = model_utils.load_model(models[model_id])

    print("\nLoading precomputed traces...")
    traces = data_utils.load_data(f"{root_folder}/lstm_input/{model_config.model_name}/{model_config.model_name}_traces.pkl")
    labels = data_utils.load_data(f"{root_folder}/lstm_input/{model_config.model_name}/{model_config.model_name}_labels.pkl")

    trained_lstm_model = train(traces, labels, num_total_experts=model_config.num_router_expert)

    ##############################################################################
    # Print some example trigger token analyses for malicious and benign prompts #
    ##############################################################################
    # questions, _ = data_utils.load_twinset(root_folder, malicious_only=True)

    # # Collect some examples of pairs in the twin set
    # for prompt_index in [0,1,2,3,4,250,251,252,253,254]:
    #     print(f"questions[prompt_index]: {questions[prompt_index]}")
    #     risk_scores = find_safety_experts.analyze_risk_trajectory(trained_lstm_model, traces, prompt_idx=prompt_index)
    #     find_safety_experts.trigger_token_analysis(tokenizer, questions[prompt_index], risk_scores)

    ##############################################################################
    ##############################################################################

    dataset = lstm_data.MoETraceDataset(traces, labels)
    data_loader = lstm_data.get_dataLoader(dataset, batch_size=1, shuffle=False)

    all_rankings = []

    for prompt_index, (x_batch, y_batch, lengths) in enumerate(data_loader):
        # if prompt_index == 260:
        #     break
        importance_map, expert_ids = find_safety_experts.get_expert_importance(trained_lstm_model, x_batch, lengths)
        rankings = find_safety_experts.find_top_refusal_experts(importance_map, expert_ids)
        # filtered_rankings = [x for x in rankings if x[1] >= 0]
        # print(f"rankings: {filtered_rankings}")
        all_rankings.append(rankings)

    accumulator = defaultdict(lambda: [0.0, 0])

    # Iterate through every list and every item
    for sublist in all_rankings:
        for key, value in sublist:
            accumulator[key][0] += value  # Add to sum
            accumulator[key][1] += 1  # Increment count

    ###################################
    ###################################

    # refusal_experts_m = []
    # for key, (total, count) in accumulator.items():
    #     refusal_experts_m.append((key, total))

    # refusal_experts_m.sort(key=lambda x: x[1], reverse=True)

    # print(f"\nRefusal experts malicious only: {refusal_experts_m}")

    # data_utils.create_directory(f"{root_folder}/refusal_experts_m")

    # data_utils.save_data(
    #     refusal_experts_m,
    #     f"{root_folder}/refusal_experts_m/{model_config.model_name}_refusal_experts_m.pkl",
    # )

    ###################################
    ###################################

    refusal_experts = []
    for key, (total, count) in accumulator.items():
        refusal_experts.append((key, total))

    refusal_experts.sort(key=lambda x: x[1], reverse=True)

    print(f"\nRefusal experts: {refusal_experts}")

    data_utils.create_directory(f"{root_folder}/refusal_experts")

    data_utils.save_data(
        refusal_experts,
        f"{root_folder}/refusal_experts/{model_config.model_name}_refusal_experts.pkl",
    )

    print("\n------------------ Job Finished ------------------")
