import numpy
from model_utils import load_model, register_activation_hooks, register_pruning_hooks, generate_output, construct_judge_prompt, moderate
from data_utils import load_datasets, expand_data, construct_prompt, batchify
from tqdm import tqdm
import torch
import pickle
from keybert import KeyBERT
import re
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import pathlib
from collections import defaultdict, Counter
from argument_parser import parse_arguments


def run_pruning():
    return None

def create_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def get_activation(model, prompts, batch_size, num_responses):
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_prompts in tqdm(batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        for _ in range(num_responses):
            with torch.no_grad():
                _ = model(**input_tokens)

    for layer_name in top_k_expert_indices:
        # print(f"Number of batches recorded in layer '{layer_name}': '{len(top_k_expert_indices[layer_name])}'")
        for batch_index, batch_expert_indices in enumerate(top_k_expert_indices[layer_name]):
            print(f"Layer: '{layer_name}', batch: '{batch_index}', shape: '{batch_expert_indices.shape}'")

def hf_custom_tokenizer(text):
    return re.findall(r'\s*\S+', text) 

def keyword_tokens_to_safety_classification(tokenizer, prompts, labels, batch_size, num_responses, top_k_expert_indices):
    safety_classifications = defaultdict(list)
    
    kw_model = KeyBERT()
    batch_size = 32
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    vectorizer = CountVectorizer(
        tokenizer=hf_custom_tokenizer,  # Custom tokenizer that returns whole words
        ngram_range=(1, 1),
        stop_words="english",
        lowercase=False  # Crucial setting to prevent lowercasing of tokens that cause mismatch with prompt tokens
    )

    # Loop over batched prompts, we need to do this batched as it allows for easy extraction of the saved routings in 'top_k_expert_indices'
    for batch_index, batch_prompts in enumerate(tqdm(batchify(prompts, batch_size), total=total_batches)):
        input_tokens = tokenizer(batch_prompts, return_tensors="np", padding=True, truncation=True)["input_ids"]
        seq_len = input_tokens.shape[1] // batch_size

        for prompt_index, prompt in enumerate(batch_prompts):  # Loop over all prompts in the batch
            # print(f"prompt: '{prompt}'")

            label = labels[(batch_index * batch_size) + prompt_index]  # Set the associated label (safe or unsafe)
            user_questions = []
            match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)  # Search for the user part of the prompt
            if match:
                user_questions.append(match.group(1))  # Append only the user part of the prompt to use for keywords search
            # print(f"user_questions: '{user_questions}'")

            keywords = kw_model.extract_keywords(user_questions, vectorizer=vectorizer)
            # print(f"keywords: '{keywords}'")
            
            tokenized_keywords = tokenizer([tuple[0] for tuple in keywords])["input_ids"]
            # print(f"tokenized_keywords: '{tokenized_keywords}'")

            tokenized_keywords = [token for sublist in tokenized_keywords for token in sublist]
            # print(f"flattened_tokenized_keywords: '{tokenized_keywords}'")

            prompt_input_tokens = input_tokens[prompt_index]
            # print(f"prompt_input_tokens: '{prompt_input_tokens}'")

            for token in tokenized_keywords:  # Loop over all tokens in keywords
                token_indices = numpy.where(prompt_input_tokens == token)[0]  # Find the indices where a token of a keyword occurs in the prompt

                for token_index in token_indices:  # Loop over all possible occurences of a keyword token in a prompt
                    # Set the index for which the token index matches in all collected token routings.
                    # E.g., token_index is '14' but this does not correspond to the 13th token in all collected routings.
                    # So we need a calculation to find the routing.
                    index_in_top_k_expert_indices = (prompt_index * seq_len) + token_index 

                    for layer in top_k_expert_indices:  # We need to loop over the layer since the tokens are processed by all gate layers
                        experts_who_processed_tokens = top_k_expert_indices[layer][prompt_index][index_in_top_k_expert_indices].tolist()  # Get the experts that processed the keyword token in all layers.
                        safety_classifications[layer + '_' + str(label)].extend(experts_who_processed_tokens)  # Associate the experts with a safe / unsafe label

    return safety_classifications

def save_dict(data, dir):
    with open(dir, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data

if __name__ == "__main__":
    arguments = parse_arguments()

    model_id = arguments.model_id
    perform_expert_hook = arguments.perform_expert_hook
    perform_safety_classifications = arguments.perform_safety_classifications

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ]

    model_name = models[model_id].split("/")[-1]

    print(f"Tested Model: '{model_name}'\n")

    model, tokenizer = load_model(models[model_id], device="auto")

    questions, labels = load_datasets() # label: 1 = unsafe
    questions, labels = expand_data(questions, labels, num_responses=1)

    prompts = construct_prompt(tokenizer, questions, system_prompt=None)

    if perform_expert_hook:
        hook_handles, top_k_expert_indices = register_activation_hooks(model)

        get_activation(model, prompts, batch_size=32, num_responses=1)

        # Remove hooks to clean up
        for handle in hook_handles:
            handle.remove()

        create_dir(f'../pre_computed_act')
        save_dict(top_k_expert_indices, f"../pre_computed_act/top_k_expert_indices_{model_name}.p")
    else:
        top_k_expert_indices = load_dict(f"../pre_computed_act/top_k_expert_indices_{model_name}.p")

    if perform_safety_classifications:
        safety_classifications = keyword_tokens_to_safety_classification(tokenizer, prompts, labels, batch_size=32, num_responses=1, top_k_expert_indices=top_k_expert_indices)

        least_picked_experts_for_unsafe_prompts = defaultdict(list)
        for layer in safety_classifications:
            if layer.endswith("1"):
                counter = Counter(safety_classifications[layer])
                least_picked_experts_for_unsafe_prompts[layer] = [key for key, _ in sorted(counter.items(), key=lambda x: x[1])[:8]]

        create_dir(f'../pre_computed_act')
        save_dict(safety_classifications, f"../pre_computed_act/safety_classifications_{model_name}.p")
        save_dict(least_picked_experts_for_unsafe_prompts, f"../pre_computed_act/least_picked_experts_for_unsafe_prompts_{model_name}.p")
    else:
        least_picked_experts_for_unsafe_prompts = load_dict(f"../pre_computed_act/least_picked_experts_for_unsafe_prompts_{model_name}.p")
        safety_classifications = load_dict(f"../pre_computed_act/safety_classifications_{model_name}.p")
        
    sorted_experts = defaultdict(list)
    for layer_name in safety_classifications:
        if layer_name.endswith("1"):
            counter = Counter(safety_classifications[layer_name])
            sorted_experts[layer_name.replace("gate_1", "gate")] = [key for key, _ in counter.items()]

    n = 100
    pruning_handles = register_pruning_hooks(model, sorted_experts, n)

    # batch_size = 32
    # total_batches = (len(prompts) + batch_size - 1) // batch_size
    # start = 0
    # num_responses = 1
    # for batch_prompts in tqdm(batchify(prompts, batch_size), total=total_batches):
    #     # Tokenize the batch
    #     # input_tokens.size = (batch_size, seq_len)
    #     input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    #     for _ in range(num_responses):
    #         with torch.no_grad():
    #             _ = model(**input_tokens)
    
    all_responses = []
    responses = generate_output(model, tokenizer, prompts, max_new_tokens=512)
    all_responses.append(responses)

    # Remove hooks to clean up
    for handle in pruning_handles:
        handle.remove()
    
    for responses in all_responses:
        print(responses)
        break

    # print("Response judgement by llama...")
    # judge_model_name = "meta-llama/Llama-Guard-3-8B"
    # judge_model, judge_tokenizer = load_model(judge_model_name, device="auto")
    # for responses in all_responses:
    #     judge_prompts = construct_judge_prompt(judge_tokenizer, questions, responses)
    #     jailbreaking_counter = 0
    #     judge_outputs = []
    #     for i, judge_prompt in enumerate(judge_prompts):
    #         judge_output = moderate(judge_model, judge_tokenizer, judge_prompt)
    #         judge_outputs.append(judge_output)
            
    #         if 'unsafe' in judge_output and not any(keyword in  str(responses[i]) for keyword in ["I cannot", "I **cannot**", "I will not", "I **will not**"]):
    #             jailbreaking_counter+=1

    #     print(f"Success rate: {jailbreaking_counter}/{len(questions)}")