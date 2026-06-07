import torch
import numpy as np
import sys
import os
import inspect
from tqdm import tqdm

# Import modules from our codebase
import argument_parser
import moe_models.model_configurations as model_configurations
import moe_models.model_utils as model_utils
import data.data_utils as data_utils


def find_token_range_by_offsets(prompt_text, question_text, offsets, print_logging):
    """Finds token start and end using character offset mappings."""
    char_start = prompt_text.rfind(question_text.strip())
    if char_start == -1:
        return None, None

    char_end = char_start + len(question_text.strip())
    start_idx = None
    end_idx = None

    for i, (tok_start, tok_end) in enumerate(offsets):
        # Skip padding or special tokens
        if tok_start == tok_end:  
            continue
        if start_idx is None and tok_end > char_start:
            start_idx = i
        if tok_start < char_end:
            end_idx = i

    if print_logging and start_idx is not None:
        print(f"Question char range: {char_start} to {char_end}")
        print(f"start token idx: {start_idx}, end token idx: {end_idx}")

    return start_idx, end_idx


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
    print(f"\nTested Model: {model_config.model_name}")

    model, tokenizer = model_utils.load_model(models[model_id])
    questions, labels = data_utils.load_twinset(root_folder, malicious_only=False)  # label 1 = refusal behavior
    prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

    current_batch_top_k = {}

    def get_top_k_hook(model_name, layer_name, k):
        def hook(module, input, output):
            # Execute your model-specific logic to extract the indices
            if model_name in [
                "Qwen3-30B-A3B-Instruct-2507",
                "Phi-3.5-MoE-instruct",
                "Mixtral-8x7B-Instruct-v0.1",
                "gpt-oss-20b",
                "Qwen1.5-MoE-A2.7B-Chat",
                "Hunyuan-A13B-Instruct",
                "OLMoE-1B-7B-0125-Instruct",
            ]:
                indices = torch.topk(output, k=k, dim=-1, sorted=False).indices
            elif model_name == "deepseek-moe-16b-chat":
                indices = output[0]
            elif model_name == "pangu-pro-moe-model":
                num_groups = 8
                experts_per_group = 8
                routing_weights, selected_experts = torch.max(output.view(output.shape[0], num_groups, -1), dim=-1)
                bias = torch.arange(
                    0,
                    64,
                    experts_per_group,
                    device=routing_weights.device,
                    dtype=torch.int64,
                ).unsqueeze(0)
                indices = selected_experts + bias
            else:
                print("---------------------------------------------------------------")
                print(input)
                print("---------------------------------------------------------------")
                print(output)
                print("---------------------------------------------------------------")
                print(output.shape)
                print("---------------------------------------------------------------")
                raise ValueError(f"Activation hook for {model_name} is not implemented.")

            # Assign to the batch dictionary, and isolate from VRAM by converting to numpy
            current_batch_top_k[layer_name] = indices.detach().cpu().numpy()

        return hook

    # Registering the Hooks
    handles = []
    layer_names = [n for n, m in model.named_modules() if n.lower().endswith(model_config.gate_name.lower())]
    for name in layer_names:
        module = dict(model.named_modules())[name]

        # Pass the model_name and k into the hook generator
        hook_fn = get_top_k_hook(model_config.model_name, name, model_config.top_k)
        handles.append(module.register_forward_hook(hook_fn))

    batch_size = 32
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    final_traces = []
    final_labels = []
    failed_matches = 0

    print(f"\nStarting Trace Collection for {len(prompts)} prompts...")

    with torch.inference_mode():
        for b_idx, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
            # Clear memory from previous batch
            current_batch_top_k.clear()

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
            offset_mappings = inputs.pop("offset_mapping").cpu().numpy()
            inputs = inputs.to(model.device)

            if "token_type_ids" in inputs:
                forward_args = inspect.signature(model.forward).parameters
                if "token_type_ids" not in forward_args:
                    inputs.pop("token_type_ids")

            b_size, s_len = inputs.input_ids.shape

            # Forward pass (triggers hooks)
            model(**inputs)

            # Safety Reshape for FLATTENED MoE outputs (e.g. OLMoE, Mixtral)
            for layer_name in layer_names:
                if current_batch_top_k[layer_name].ndim == 2:
                    current_batch_top_k[layer_name] = current_batch_top_k[layer_name].reshape(b_size, s_len, -1)

            # Process the batch data immediately
            for p_idx in range(b_size):
                global_p_idx = (b_idx * batch_size) + p_idx
                if global_p_idx >= len(prompts):
                    break

                prompt_text = batch_prompts[p_idx]
                q_text = questions[global_p_idx]

                if print_logging and b_idx == 0 and p_idx == 0:
                    print(f"\nExample Prompt:\n{prompt_text}")

                # Find exact indices using offset mapping (no model-specific string hacks needed)
                start, end = find_token_range_by_offsets(
                    prompt_text, q_text, offset_mappings[p_idx], print_logging=(print_logging and b_idx == 0 and p_idx == 0)
                )

                if start is None:
                    failed_matches += 1
                    if print_logging:
                        print(f"\nFailed to find token match for prompt index {global_p_idx}")
                    continue

                prompt_trace = []

                # Slice and append data for each layer for THIS specific prompt
                for layer_name in layer_names:
                    # current_batch_top_k[layer_name] shape: (batch, seq_len, top_k)
                    sliced_indices = current_batch_top_k[layer_name][p_idx, start : end + 1, :]
                    prompt_trace.append(sliced_indices)

                # Stack layer traces into a single 3D array: (seq_length, num_layers, top_k)
                stacked_trace = np.stack(prompt_trace, axis=1)

                final_traces.append(stacked_trace)
                final_labels.append(labels[global_p_idx])

    # Cleanup hooks
    for h in handles:
        h.remove()

    if failed_matches > 0:
        print(f"\nWarning: Could not find exact token match for {failed_matches} prompts.")

    # Save outputs
    output_dir = os.path.join(root_folder, "data", "lstm_input", model_config.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving traces to disk...")
    data_utils.save_data(final_traces, os.path.join(output_dir, f"{model_config.model_name}_traces.pkl"))
    data_utils.save_data(final_labels, os.path.join(output_dir, f"{model_config.model_name}_labels.pkl"))

    if print_logging and len(final_traces) > 0:
        print(f"\nNumber of traces: {len(final_traces)}")
        print(f"Number of labels: {len(final_labels)}")
        print(f"Shape of a single trace (seq_len, layers, top_k): {final_traces[0].shape}")

        # Print a small slice of the very first token to verify structure
        print(f"\nTrace [0] (Token 0, All Layers, Top K experts):\n{final_traces[0][0, :, :]}")
        print(f"labels[0]: {final_labels[0]}")

    print("\n------------------ Job Finished ------------------")
