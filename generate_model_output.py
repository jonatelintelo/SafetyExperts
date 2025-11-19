import model_utils
import data_utils
from collections import defaultdict, Counter
from transformers.cache_utils import DynamicCache

DynamicCache.get_max_length = DynamicCache.get_seq_length

BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def generate_model_output():
    model_id = 0
    dry_run = False

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "deepseek-ai/deepseek-moe-16b-chat",
        "microsoft/Phi-3.5-MoE-instruct",
        "openai/gpt-oss-20b",
    ]

    n_experts = {
        "Qwen3-30B-A3B-Instruct-2507": 128,
        "deepseek-moe-16b-chat": 64,
        "Phi-3.5-MoE-instruct": 16,
        "gpt-oss-20b": 32,
    }

    top_k = {
        "Qwen3-30B-A3B-Instruct-2507": 8,
        "deepseek-moe-16b-chat": 6,
        "Phi-3.5-MoE-instruct": 2,
        "gpt-oss-20b": 4,
    }

    model_name = models[model_id].split("/")[-1]

    model, tokenizer = model_utils.load_model(models[model_id], device="auto")

    total_number_of_layers = 0
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            total_number_of_layers += 1

    n_expert = n_experts[model_name]
    
    print(f"Tested Model: '{model_name}'")
    print(f"\nTotal number of gate layers in '{model_name}': '{total_number_of_layers}'")
    print(f"\nNumber of experts in '{model_name}': '{n_expert}'")

    iter_list = [0.1,0.2,0.3]

    if dry_run:
        iter_list = [0]

    progessive_strategy = True

    if not progessive_strategy:
        total_number_of_layers = 1

    for number_of_layers in range(1, total_number_of_layers + 1):
    # for number_of_layers in range(14, 0, -1):

        for p in iter_list:
            n = round(p*n_expert)
            print(f"\nNumber of gate layers which will be pruned: '{number_of_layers}'")
            print(f"Number of experts to be pruned: '{n}({p*100}%)'", flush=True)

            if n > 0:
                if not progessive_strategy:
                    number_of_layers = 'all'
                    pruning_handles = model_utils.register_pruning_hooks(model_name, model, sorted_experts, n, k=top_k[model_name])
                else:
                    pruning_handles = model_utils.register_progressive_pruning_hooks(model_name, model, n, number_of_layers=number_of_layers, k=top_k[model_name], p=p)


            # load malicious questions
            dataset = data_utils.load_dataset("walledai/StrongREJECT")
            questions = dataset['train']['prompt']
            prompts = data_utils.construct_prompt(tokenizer, questions, model_name)
            
            # batch_size = 1 if "deepseek" in model_name else 8
            responses = model_utils.generate_output(model, model_name, tokenizer, prompts, max_new_tokens=256, batch_size=256)
            
            data_utils.create_dir(BASE_FOLDER + f"precomputed_responses/{model_name}")

            data_utils.save_data(responses, BASE_FOLDER + f"precomputed_responses/{model_name}/{model_name}_responses_{number_of_layers}_{p}.pkl")
            
            if n > 0:
                # Remove hooks to clean up
                for handle in pruning_handles:
                    handle.remove()

    print("\n------------------ Job Finished ------------------")

if __name__ == "__main__":
    generate_model_output()

