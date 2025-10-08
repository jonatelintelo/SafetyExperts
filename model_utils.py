import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy
from data_utils import batchify
import re
from tqdm import tqdm


def load_model(model_name, device):
    attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_implementation,
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
        # responses = [clean_generated_text(tokenizer.decode(generated_tokens[i], skip_special_tokens=True)) for i in range(len(generated_tokens))]
        print(responses, flush=True)
        all_outputs.extend(responses)

    return all_outputs

def clean_generated_text(text):
    """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
    # if model_name.startswith("DeepSeek"):
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
    return response  # No </think> tag, return entire response

def construct_judge_prompt(tokenizer, questions, responses):
    prompts = []
    for question, response in zip(questions, responses):
        if not response.strip():
            response = "Sorry, I cannot assist with that."
        response = extract_text_after_think(response)
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(response).replace('[', '').replace(']', '')},
        ]
        # prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(chat)
    return prompts

def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def get_activation_hook(layer_name, top_k_expert_indices):
    def activation_hook(module, input, output):
        # sequence_lengths.append(input[0].shape[0] // 32)
        # print(f"{layer_name} shape of topk: '{numpy.array(torch.topk(output, k=8, dim=-1).indices.cpu()).shape}'")
        top_k_expert_indices[layer_name].append(torch.topk(output, k=8, dim=-1).indices.cpu())  # (nr_tokens, topk_experts)
        # print(f"top_k_expert_indices[layer_name]: '{top_k_expert_indices[layer_name]}'")
        # print(f"shape of top_k_expert_indices[layer_name]: '{numpy.array(top_k_expert_indices[layer_name]).shape}'")
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

# def get_expert_routing_hook(layer_name, candidate_neurons):
#     def expert_routing_hook(module, input, output):
#         print(f"Routing tokens to different experts in layer: '{layer_name}'")

#         # output shape: [batch, seq_length, hidden_dim]
#         pruned_output = output.clone()
#         pruned_output[..., candidate_neurons] = 0  # Zero out the specified neurons
#         return pruned_output
#     return expert_routing_hook

# def register_expert_routing_hooks(model, prioritized_experts, target_layer):
#     hook_handles = []

#     for layer_name, module in model.named_modules():
#         if layer_name.lower().endswith(".mlp.gate"):
#             hook_fn = get_expert_routing_hook(layer_name, prioritized_experts)
#             handle = module.register_forward_hook(hook_fn)
#             hook_handles.append(handle)
    
#     return hook_handles

def get_prune_hook(layer_name, sorted_experts):
    def prune_hook(module, input, output):
        # output shape: [tokens, num_experts]
        # print(f"Pruning logits of experts in: '{layer_name}'")
        for expert in sorted_experts:
            pruned_output = output.clone()
            pruned_output[...,expert] = torch.min(pruned_output, dim=1).values
        return pruned_output
    return prune_hook

# Function to register pruning hooks for all candidate layers
def register_pruning_hooks(model, sorted_experts, n):
    hook_handles = []

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith("mlp.gate"):
            hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
    
    return hook_handles