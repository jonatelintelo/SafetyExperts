from datasets import load_dataset
import pathlib
import pickle


def save_data(data, dir):
    with open(dir, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data

def create_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

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

    print(f"\nNumber of malicious prompts: '{len(all_texts)}'")
    
    if not malicious_only:
        # Benign dataset
        ds = load_dataset("facebook/natural_reasoning")
        all_texts += ds["train"]["question"][:len(all_labels)]
        all_labels += [0] * len(all_labels)
    
    print(f"Number of all prompts: '{len(all_texts)}'")
    
    return all_texts, all_labels

def construct_prompt(tokenizer, questions, model_name):
    prompts = []
    system_prompt = "You are an AI that provides direct and precise answers. Respond only to the question without additional details or explanations."

    for question in questions:
        if model_name == "deepseek-moe-16b-chat":
            chat = [
                {"role": "user", "content": question}
            ]
        if model_name in ["Phi-3.5-MoE-instruct","Qwen3-30B-A3B-Instruct-2507"]:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        else:
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