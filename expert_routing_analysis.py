import numpy
import model_utils
import data_utils
from tqdm import tqdm
import torch
import pickle
# from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import pathlib
from collections import defaultdict, Counter
from argument_parser import parse_arguments
from transformers.cache_utils import DynamicCache

DynamicCache.get_max_length = DynamicCache.get_seq_length

BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def get_activation(model, prompts, batch_size, num_responses):
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    # start = 0

    for batch_prompts in tqdm(data_utils.batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # print(f"input_tokens.shape: '{input_tokens['input_ids'].shape}'")
        
        for _ in range(num_responses):
            with torch.no_grad():
                _ = model(**input_tokens)
        # break
        # start += 1
        # if start == 3:
        #     break
    # for layer_name in top_k_expert_indices:
    #     print(f"Number of batches recorded in layer '{layer_name}': '{len(top_k_expert_indices[layer_name])}'")
    #     for batch_index, batch_expert_indices in enumerate(top_k_expert_indices[layer_name]):
    #         print(f"Layer: '{layer_name}', batch: '{batch_index}', shape: '{batch_expert_indices.shape}'")

    # dd


def tokens_to_safety_classification(tokenizer, prompts, questions, labels, model_name, batch_size, top_k_expert_indices):
    safety_classifications = defaultdict(list)
    
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    # start = 0
    # Loop over batched prompts, we need to do this batched as it allows for easy extraction of the saved routings in 'top_k_expert_indices'
    for batch_index, batch_prompts in enumerate(tqdm(data_utils.batchify(prompts, batch_size), total=total_batches)):
        input_tokens = tokenizer(batch_prompts, return_tensors="np", padding=True, truncation=True)["input_ids"]
        seq_len = input_tokens.shape[1]
        # print(f"input_tokens.shape: '{input_tokens.shape}'")

        for prompt_index, prompt in enumerate(batch_prompts):  # Loop over all prompts in the batch
            label = labels[(batch_index * batch_size) + prompt_index]  # Set the associated label (safe or unsafe)
            tokenized_prompt = tokenizer(questions[(batch_index * batch_size) + prompt_index])["input_ids"]
            prompt_input_tokens = input_tokens[prompt_index]

            # print(f"prompt: '{prompt}'")
            # print(f"questions: '{questions[(batch_index * batch_size) + prompt_index]}'")
            # print(f"tokenized_prompt: '{tokenized_prompt}'")
            # print(f"prompt_input_tokens: '{prompt_input_tokens}'")

            for token in tokenized_prompt:  # Loop over all tokens in the prompt
                token_indices = numpy.where(prompt_input_tokens == token)[0]  # Find the indices where the token occurs in the prompt

                for token_index in token_indices:  # Loop over all possible occurences of a keyword token in a prompt
                    # Set the index for which the token index matches in all collected token routings.
                    # E.g., token_index is '14' but this does not correspond to the 13th token in all collected routings.
                    # So we need a calculation to find the routing.
                    index_in_top_k_expert_indices = (prompt_index * seq_len) + token_index 

                    for layer in top_k_expert_indices:  # We need to loop over the layer since the tokens are processed by all gate layers
                        experts_who_processed_token = top_k_expert_indices[layer][index_in_top_k_expert_indices].tolist()  # Get the experts that processed the keyword token in all layers.
                        safety_classifications[layer + '_' + str(label)].extend(experts_who_processed_token)  # Associate the experts with a safe / unsafe label
                        # print(f"token: '{token}'")
                        # print(f"token_indices: '{token_indices}'")
                        # print(f"token_index: '{token_index}'")
                        # print(f"prompt_index: '{prompt_index}'")
                        # print(f"seq_len: '{seq_len}'")
                        # print(f"index_in_top_k_expert_indices: '{index_in_top_k_expert_indices}'")
                        # print(f"layer: '{layer}'")
                        # print(f"top_k_expert_indices[layer].shape: '{numpy.array(top_k_expert_indices[layer]).shape}'")
                        # print(f"top_k_expert_indices[layer]: '{top_k_expert_indices[layer]}'")
                        # print(f"experts_who_processed_token: '{experts_who_processed_token}'")
                        # dd
        #                 break
        # start += 1
        # if start == 2:
        #     dd
    return safety_classifications

# def hf_custom_tokenizer(text):
#     return re.findall(r'\s*\S+', text) 
# def keyword_tokens_to_safety_classification(tokenizer, prompts, labels, batch_size, top_k_expert_indices):
#     kw_model = KeyBERT()
#     vectorizer = CountVectorizer(
#         tokenizer=hf_custom_tokenizer,  # Custom tokenizer that returns whole words
#         ngram_range=(1, 1),
#         stop_words="english",
#         lowercase=False  # Crucial setting to prevent lowercasing of tokens that cause mismatch with prompt tokens
#     )
#     keywords = kw_model.extract_keywords(user_questions, vectorizer=vectorizer)
#     tokenized_keywords = tokenizer([tuple[0] for tuple in keywords])["input_ids"]

if __name__ == "__main__":
    # arguments = parse_arguments()

    model_id = 0

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

    print(f"Tested Model: '{model_name}'")

    model, tokenizer = model_utils.load_model(models[model_id], device="auto")

    total_number_of_layers = 0
    for layer_name, module in model.named_modules():
        if layer_name.lower().endswith(".mlp.gate") or layer_name.lower().endswith("moe.gate") or layer_name.lower().endswith(".router"):
            total_number_of_layers += 1
    
    print(f"\nTotal number of gate layers in '{model_name}': '{total_number_of_layers}'")

    questions, labels = data_utils.load_datasets() # label: 1 = unsafe
    prompts = data_utils.construct_prompt(tokenizer, questions, model_name)

    n_expert = n_experts[model_name]

    print(f"\nNumber of experts in '{model_name}': '{n_expert}'")

    iter_list = [0.1,0.2,0.3]
    batch_size = 32

    progessive_strategy = False

    for number_of_layers in range(1, total_number_of_layers + 1):
        for p in iter_list:
            n = round(p*n_expert)
            if not progessive_strategy:
                number_of_layers = 'all'
                n = 'all'
                p = 'all'
            print("\nPerforming expert hooking and activation analysis...")
            print(f"Number of gate layers which will be pruned: '{number_of_layers}'")
            print(f"Number of experts to be pruned: '{n}({p*100 if progessive_strategy else 'all'}%)'\n")

            if not progessive_strategy:
                number_of_layers = 'all'
                hook_handles, top_k_expert_indices = model_utils.register_activation_hooks(model_name, model, k=top_k[model_name])
            else:
                hook_handles, top_k_expert_indices = model_utils.register_progressive_strategy_hooks(model_name, model, k=top_k[model_name], n=n, number_of_layers=number_of_layers, p=p)

            get_activation(model, prompts, batch_size=batch_size, num_responses=1)

            # Remove hooks to clean up
            for handle in hook_handles:
                handle.remove()
            
            # Concatenate activations for each layer.
            for layer_name in tqdm(top_k_expert_indices):
                top_k_expert_indices[layer_name] = numpy.concatenate(top_k_expert_indices[layer_name], axis=0)

            data_utils.create_dir(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}")

            data_utils.save_data(top_k_expert_indices, BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_expert_indices_{number_of_layers}_{p}.pkl")

            print("\nPerforming safety classification...")

            print("\nLoading precomputed activation analysis...")
            top_k_expert_indices = data_utils.load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_expert_indices_{number_of_layers}_{p}.pkl")
            
            safety_classifications = tokens_to_safety_classification(tokenizer, prompts, questions, labels, model_name, batch_size=batch_size, top_k_expert_indices=top_k_expert_indices)
        
            sorted_experts = defaultdict(list)
            
            for layer_name in safety_classifications:
                if layer_name.endswith("1"):
                    counter = Counter(safety_classifications[layer_name])
                    sorted_experts[layer_name.replace("_1", "")] = [key for key, _ in counter.most_common()]

            data_utils.save_data(sorted_experts, BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_sorted_experts_{number_of_layers}_{p}.pkl")
            
            if not progessive_strategy:
                data_utils.save_data(safety_classifications, BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_safety_classifications_{number_of_layers}_{n}.pkl")
                break

        if not progessive_strategy:
            break
        
    print("\n------------------ Job Finished ------------------")
    
        # keyword_safety_classifications = keyword_tokens_to_safety_classification(tokenizer, prompts, labels, batch_size=32, top_k_expert_indices=top_k_expert_indices)
        # keyword_sorted_experts = defaultdict(list)
        # for layer_name in keyword_safety_classifications:
        #     if layer_name.endswith("1"):
        #         keyword_counter = Counter(keyword_safety_classifications[layer_name])
        #         keyword_sorted_experts[layer_name.replace("gate_1", "gate")] = [key for key, _ in keyword_counter.most_common()]
        # data_utils.save_data(keyword_safety_classifications, BASE_FOLDER + f"precomputed_expert_analysis/keyword_safety_classifications_{model_name}.pkl")
        # data_utils.save_data(keyword_sorted_experts, BASE_FOLDER + f"precomputed_expert_analysis/keyword_sorted_experts_{model_name}.pkl")
    
    #     else:
    #         print("\nLoading pre-computed safety classification...")
    #         keyword_safety_classifications = data_utils.load_data(BASE_FOLDER + f"pre_computed_act/keyword_safety_classifications_{model_name}.p")
    #         keyword_sorted_experts = data_utils.load_data(BASE_FOLDER + f"pre_computed_act/keyword_sorted_experts_{model_name}.p")
    #         safety_classifications = data_utils.load_data(BASE_FOLDER + f"pre_computed_act/safety_classifications_{model_name}.p")
    #         sorted_experts = data_utils.load_data(BASE_FOLDER + f"pre_computed_act/sorted_experts_{model_name}.p")