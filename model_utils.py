import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from collections import defaultdict
import numpy
from collections import defaultdict, Counter
import data_utils
import re
from tqdm import tqdm
BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=False,
        # attn_implementation="flash_attention_2"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_output(model, model_name, tokenizer, prompts, batch_size, max_new_tokens):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    # if model_name in ["Phi-3.5-MoE-instruct","gpt-oss-20b"]:
    pipe = pipeline( 
        "text-generation", 
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )

    model.eval()
    with torch.no_grad():
        # if model_name in ["Phi-3.5-MoE-instruct","gpt-oss-20b"]:
        outputs = pipe(prompts)
        for i, output in enumerate(outputs):
            # The output is often a list of dictionaries, extract the generated text
            response = output[0]["generated_text"].split('assistantfinal', 1)
            if len(response) > 1:
                response = response[1]
            else:
                response = response[0]
            all_outputs.append(response)
            # print(f"\n--- Prompt {(batch_index*batch_size) + i + 1} ---")
            print(f"\n--- Prompt {i + 1} ---")
            # print(f"output: '{output}'")
            print(f"response: '{response}'")

    return all_outputs

    #     for batch_index, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
    #         # Tokenize the batch
    #         input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
    #         input_ids = input_tokens["input_ids"]

    #         output_ids = model.generate(**input_tokens,
    #                                     max_new_tokens=max_new_tokens,
    #                                     return_dict_in_generate=True
    #         )

    #         generated_tokens = [
    #             output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])
    #         ]

    #         # Decode the generated outputs
    #         responses = [clean_generated_text(tokenizer.decode(generated_tokens[i], skip_special_tokens=True)) for i in range(len(generated_tokens))]
    #         all_outputs.extend(responses)

    #         for i, response in enumerate(responses):
    #             print(f"\n--- Prompt {(batch_index*batch_size) + i + 1} ---")
    #             print(f"response: '{response}'")
    #         # if batch_index == 2:
    #         #     break
            
    # return all_outputs

def get_activation_hook(model_name, layer_name, top_k_expert_indices, k):
    print(f"Activation hook on layer: '{layer_name}'")

    def activation_hook(module, input, output):
        if "deepseek" in model_name:
            top_k_expert_indices[layer_name].append(output[0].cpu())
        if model_name == "gpt-oss-20b":
            top_k_expert_indices[layer_name].append(output[1].cpu())
        else:
            # print("---------------------------------------------------------------")
            # print(input)
            # print("---------------------------------------------------------------")
            # print(output[0])
            # print("---------------------------------------------------------------")
            # print(torch.topk(output, k=k, dim=-1).indices.cpu().shape)
            # dd
            top_k_expert_indices[layer_name].append(torch.topk(output, k=k, dim=-1).indices.cpu())  # (nr_tokens, topk_experts)
    return activation_hook

def register_progressive_strategy_hooks(model_name, model, k, n, number_of_layers, p):
    hook_handles = []
    top_k_expert_indices = defaultdict(list)
    number_of_processed_layers = 0

    # is 0 == 2? no
    # load 1_n and prune moe layer 1 
    # number_of_processed_layers == 1

    # is 1 == 2? no
    # load 2_n and prune moe layer 2
    # number_of_processed_layers == 2

    # is 2 == 2? yes
    # activation on moe layer 3 
    # number_of_processed_layers == 3
    
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            if number_of_processed_layers == number_of_layers - 1:
                hook_fn = get_activation_hook(model_name, layer_name, top_k_expert_indices, k)
                number_of_processed_layers += 1
            else:
                sorted_experts = data_utils.load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_sorted_experts_{number_of_processed_layers+1}_{p}.pkl")
                if "deepseek" in model_name or model_name == "gpt-oss-20b":
                    hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
                else:
                    hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
                number_of_processed_layers += 1

            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

        if number_of_processed_layers == number_of_layers:
            return hook_handles, top_k_expert_indices

def register_progressive_pruning_hooks(model_name, model, n, number_of_layers, k, p):
    hook_handles = []
    number_of_processed_layers = 0
    
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            sorted_experts = data_utils.load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_sorted_experts_{number_of_processed_layers+1}_{p}.pkl")

            if "deepseek" in model_name or model_name == "gpt-oss-20b":
                hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
            else:
                hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
                
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
        
            number_of_processed_layers += 1

        if number_of_processed_layers == number_of_layers:
            return hook_handles

def register_activation_hooks(model_name, model, k):
    hook_handles = []
    top_k_expert_indices = defaultdict(list)

    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            hook_fn = get_activation_hook(model_name, layer_name, top_k_expert_indices, k)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles, top_k_expert_indices

def get_prune_hook(layer_name, sorted_experts):
    print(f"Pruning hook on layer: '{layer_name}'")

    def prune_hook(module, input, output):
        pruned_output = output.clone()
        pruned_output[...,sorted_experts] = float("-inf")

        return pruned_output

    return prune_hook

def get_routing_hook(layer_name, sorted_experts, model_name, k):
    print(f"Routing hook on layer: '{layer_name}'")

    if model_name == "gpt-oss-20b":
        def routing_hook(module, input, output):
            pruned_output = tuple(t.clone() if t is not None else None for t in output)
            pruned_output[0][...,sorted_experts] = float("-inf")
            pruned_output[1][:] = torch.topk(pruned_output[0], k=k, dim=-1, sorted=False).indices
            return pruned_output
    else:   
        def routing_hook(module, input, output):
            pruned_output = tuple(t.clone() if t is not None else None for t in output)
            pruned_output[3][...,sorted_experts] = float("-inf")
            pruned_output[1][:], pruned_output[0][:] = torch.topk(pruned_output[3], k=6, dim=-1, sorted=False)
            return pruned_output

    return routing_hook

def register_pruning_hooks(model_name, model, sorted_experts, n, k):
    hook_handles = []
   
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            if "deepseek" in model_name or model_name == "gpt-oss-20b":
                hook_fn = get_routing_hook(layer_name, sorted_experts[layer_name][:n], model_name, k)
            else:
                hook_fn = get_prune_hook(layer_name, sorted_experts[layer_name][:n])
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    return hook_handles
    
def construct_judge_prompt(tokenizer, questions, responses):
    prompts = []
    for question, response in zip(questions, responses):
        if not response.strip():
            response = "Sorry, I cannot assist with that."
        response = extract_text_after_think(response)
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(response).replace("[", "").replace("]", "")},
        ]
        prompts.append(chat)
    return prompts

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

def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)