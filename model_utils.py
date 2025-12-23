import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from collections import defaultdict
import data_utils
import re
from tqdm import tqdm
BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def load_model(model_name):
    if model_name in ["microsoft/Phi-3.5-MoE-instruct"]:
        trust_remote_code = False
    else:
        trust_remote_code = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_output(model, model_name, tokenizer, prompts, batch_size, max_new_tokens):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    with torch.no_grad():
        for batch_prompts in tqdm(data_utils.batchify(prompts, batch_size), total=total_batches):
            input_tokens = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(model.device)

            if "token_type_ids" in input_tokens:
                print("token_type_ids FOUND")

            input_ids = input_tokens["input_ids"]

            output_ids = model.generate(
                **input_tokens, 
                max_new_tokens=max_new_tokens, 
                return_dict_in_generate=True,
                max_length=max_new_tokens
            )

            generated_tokens = [
                output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])
            ]

            responses = [tokenizer.decode(generated_tokens[i], skip_special_tokens=True) for i in range(len(generated_tokens))]

            all_outputs.extend(responses)

    return all_outputs

def clean_generated_text(text):
    """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
    text = extract_text_after_think(text)
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    text = re.sub(r"^(assistant\n|Assistant:|AI:|Bot:|Response:|Reply:|.:)\s*", "", text, flags=re.IGNORECASE)  # Remove AI labels if present
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    return text

def extract_text_after_think(response: str) -> str:
    # Find all occurrences of </think>
    think_matches = list(re.finditer(r"</think>", response))

    if think_matches:
        # Get the last occurrence
        last_think_index = think_matches[-1].end()
        return response[last_think_index:].lstrip()  # Strip leading spaces/newlines
    else:
        return response  # No </think> tag, return entire response

def get_activation_hook(model_name, layer_name, top_k_expert_indices, k):
    print(f"Activation hook on layer: '{layer_name}'")

    def activation_hook(module, input, output):
        if model_name in ["Phi-3.5-MoE-instruct","Qwen3-30B-A3B-Instruct-2507"]:
            top_k_expert_indices[layer_name].append(torch.topk(output, k=k, dim=-1).indices.cpu())  # (nr_tokens, topk_experts)
        else:
            print("---------------------------------------------------------------")
            print(input)
            print("---------------------------------------------------------------")
            print(output)
            print("---------------------------------------------------------------")
            print(output.shape)
            print("---------------------------------------------------------------")
            raise ValueError(f"Activation hook for {model_name} is not implemented.")
    return activation_hook

def register_activation_hooks(model_name, model, k, gate_name):
    hook_handles = []
    top_k_expert_indices = defaultdict(list)

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            hook_fn = get_activation_hook(model_name, layer_name, top_k_expert_indices, k)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles, top_k_expert_indices

def get_pruning_hook(layer_name, refusal_drivers):
    # print(f"Pruning hook on layer: '{layer_name}'")
    def prune_hook(module, input, output):
        pruned_output = output.clone()
        pruned_output[...,refusal_drivers] = float("-inf")
        return pruned_output

    return prune_hook

def get_pruning_hook_on_experts(layer_name, experts):
    # print(f"Pruning hook on layer: '{layer_name}'")
    def prune_hook(module, input, output):
        pruned_output = output.clone()
        pruned_output[...,experts] = float("-inf")
        return pruned_output

    return prune_hook

def get_pruning_hook_candidates(layer_name, expert_index):
    # print(f"Pruning hook on layer: '{layer_name}' and expert '{expert_index}'")

    def prune_hook(module, input, output):
        pruned_output = output.clone()
        # print(f"output.shape: {output.shape}")
        # before = torch.topk(pruned_output, k=2, dim=-1, sorted=False).indices
        pruned_output[...,expert_index] = float("-inf")
        # print(before == torch.topk(pruned_output, k=2, dim=-1, sorted=False).indices)
        return pruned_output

    return prune_hook

def register_pruning_hooks_candidates(model_name, model, candidates):
    hook_handles = []

    for candidate in candidates:
        for layer_name, module in model.named_modules():
            if layer_name.lower().endswith(candidate[0]):
                hook_fn = get_pruning_hook(layer_name, candidate[1])
                handle = module.register_forward_hook(hook_fn)
                hook_handles.append(handle)
                break

    return hook_handles

def register_pruning_hooks(model, refusal_drivers, gate_name):
    hook_handles = []

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            layer_index = int(layer_name.split('.')[2])
            hook_fn = get_pruning_hook(layer_name, refusal_drivers[layer_index])
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles

def register_pruning_hooks_on_experts(model, experts, gate_name):
    hook_handles = []

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(gate_name):
            hook_fn = get_pruning_hook_on_experts(layer_name, experts)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles


# def register_progressive_strategy_hooks(model_name, model, k, n, number_of_layers, p):
#     hook_handles = []
#     top_k_expert_indices = defaultdict(list)
#     number_of_processed_layers = 0

#     # is 0 == 2? no
#     # load 1_n and prune moe layer 1 
#     # number_of_processed_layers == 1

#     # is 1 == 2? no
#     # load 2_n and prune moe layer 2
#     # number_of_processed_layers == 2

#     # is 2 == 2? yes
#     # activation on moe layer 3 
#     # number_of_processed_layers == 3
    
#     for layer_name, module in model.named_modules():
#         if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
#             if number_of_processed_layers == number_of_layers - 1:
#                 hook_fn = get_activation_hook(model_name, layer_name, top_k_expert_indices, k)
#                 number_of_processed_layers += 1
#             else:
#                 sorted_experts = data_utils.load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_sorted_experts_{number_of_processed_layers+1}_{p}.pkl")
#                 if "deepseek" in model_name or model_name == "gpt-oss-20b":
#                     hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
#                 else:
#                     hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
#                 number_of_processed_layers += 1

#             handle = module.register_forward_hook(hook_fn)
#             hook_handles.append(handle)

#         if number_of_processed_layers == number_of_layers:
#             return hook_handles, top_k_expert_indices

# def register_progressive_pruning_hooks(model_name, model, n, number_of_layers, k, p):
#     hook_handles = []
#     number_of_processed_layers = 0
    
#     for layer_name, module in model.named_modules():
#         if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
#             sorted_experts = data_utils.load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_sorted_experts_{number_of_processed_layers+1}_{p}.pkl")

#             if "deepseek" in model_name or model_name == "gpt-oss-20b":
#                 hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
#             else:
#                 hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
                
#             handle = module.register_forward_hook(hook_fn)
#             hook_handles.append(handle)
        
#             number_of_processed_layers += 1

#         if number_of_processed_layers == number_of_layers:
#             return hook_handles


# def get_prune_hook(layer_name, sorted_experts):
#     print(f"Pruning hook on layer: '{layer_name}'")

#     def prune_hook(module, input, output):
#         pruned_output = output.clone()
#         pruned_output[...,sorted_experts] = float("-inf")
#         return pruned_output

#     return prune_hook

# def get_routing_hook(layer_name, sorted_experts, model_name, k):
#     print(f"Routing hook on layer: '{layer_name}'")

#     if model_name == "gpt-oss-20b":
#         def routing_hook(module, input, output):
#             pruned_output = tuple(t.clone() if t is not None else None for t in output)
#             pruned_output[0][...,sorted_experts] = float("-inf")
#             pruned_output[1][:] = torch.topk(pruned_output[0], k=k, dim=-1, sorted=False).indices
#             return pruned_output
#     else:   
#         def routing_hook(module, input, output):
#             pruned_output = tuple(t.clone() if t is not None else None for t in output)
#             pruned_output[3][...,sorted_experts] = float("-inf")
#             pruned_output[1][:], pruned_output[0][:] = torch.topk(pruned_output[3], k=6, dim=-1, sorted=False)
#             return pruned_output

#     return routing_hook

# def register_pruning_hooks(model_name, model, sorted_experts, n, k):
#     hook_handles = []
   
#     for layer_name, module in model.named_modules():
#         if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
#             if "deepseek" in model_name or model_name == "gpt-oss-20b":
#                 hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
#             else:
#                 hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
#             handle = module.register_forward_hook(hook_fn)
#             hook_handles.append(handle)

#     return hook_handles
