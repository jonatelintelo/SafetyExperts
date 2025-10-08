import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy
from data_utils import batchify
import re
from tqdm import tqdm


def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation='flash_attention_2',
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_output(model, tokenizer, prompts, batch_size=8, max_new_tokens=1024):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_prompts in tqdm(batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = input_tokens["input_ids"]
        output_ids = model.generate(**input_tokens,
                                    max_new_tokens=max_new_tokens,
                                    return_dict_in_generate=True
        )

        # Extract only the generated tokens (excluding the input prompts)
        generated_tokens = [output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])]
        responses = ([tokenizer.decode(generated_tokens[i], skip_special_tokens=True) for i in range(len(generated_tokens))])
        # Decode the generated outputs
        for response in responses:
            print(response, flush=True)
        all_outputs.extend(responses)

    return all_outputs

def get_activation_hook(layer_name, top_k_expert_indices):
    def activation_hook(module, input, output):
        top_k_expert_indices[layer_name].append(torch.topk(output, k=8, dim=-1).indices.cpu())  # (nr_tokens, topk_experts)
    return activation_hook

def register_activation_hooks(model):
    hook_handles = []
    top_k_expert_indices = defaultdict(list)

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate"):
            hook_fn = get_activation_hook(layer_name, top_k_expert_indices)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
    return hook_handles, top_k_expert_indices

def get_prune_hook(layer_name, sorted_experts):
    def prune_hook(module, input, output):
        pruned_output = output.clone()
        pruned_output[...,sorted_experts] = float('-inf')
        return pruned_output
    return prune_hook

def register_pruning_hooks(model, sorted_experts, n):
    hook_handles = []

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith("mlp.gate"):
            hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
    return hook_handles