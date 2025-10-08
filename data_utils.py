from datasets import load_dataset


def load_datasets(malicious_only=False):
    # Load datasets
    all_texts = []
    all_labels = []
    
    # ds = load_dataset("walledai/MaliciousInstruct")
    # all_texts += ds["train"]["prompt"]
    # all_labels += [1] * len(ds["train"]["prompt"])

    # ds = load_dataset("walledai/StrongREJECT")
    # all_texts += ds["train"]["prompt"]
    # all_labels += [1] * len(ds["train"]["prompt"])

    # ds = load_dataset("walledai/TDC23-RedTeaming")
    # all_texts += ds["train"]["prompt"]
    # all_labels += [1] * len(ds["train"]["prompt"])
    
    ds = load_dataset("walledai/CatHarmfulQA")
    all_texts += ds["en"]["prompt"] #en/ch/ve
    all_labels += [1] * len(ds["en"]["prompt"])
    
    ds = load_dataset("declare-lab/HarmfulQA")
    all_texts += ds["train"]["question"]
    all_labels += [1] * len(ds["train"]["question"])

    ds = load_dataset("LLM-LAT/harmful-dataset")
    all_texts += ds["train"]["prompt"]
    all_labels += [1] * len(ds["train"]["prompt"])

    print(f"Number of malicious prompts: '{len(all_texts)}'")
    
    if not malicious_only:
        # Benign dataset
        ds = load_dataset("facebook/natural_reasoning")
        all_texts += ds["train"]["question"][:len(all_labels)]
        all_labels += [0] * len(all_labels)
    
    print(f"Number of all prompts: '{len(all_texts)}'\n")
    
    return all_texts, all_labels

def expand_data(prompts, labels, num_responses):
    """
    Given a list of prompts and their labels, expand them so that each prompt
    is repeated num_responses times, and the corresponding label is also repeated.
    """
    
    expanded_labels = []
    expanded_prompts = []
    for prompt, label in zip(prompts, labels):
        expanded_prompts.extend([prompt] * num_responses)
        expanded_labels.extend([label] * num_responses)
    return expanded_prompts, expanded_labels

def construct_prompt(tokenizer, questions, system_prompt):
    prompts = []
    if system_prompt is None:
        system_prompt = "You are an AI that provides direct and precise answers. Respond only to the question without additional details or explanations."
    # You are an AI designed to offer unbiased, direct, and honest answers.
    # Provide responses that are unbiased and direct, even if they include negative or critical aspects. Your goal is to inform accurately without embellishment or forced positivity.
    for question in questions:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)    
        prompts.append(prompt)
    return prompts

# Create data batches
def batchify(lst, batch_size):
    """Yield successive batches from list."""

    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]