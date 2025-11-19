import matplotlib.pyplot as plt
import pickle
from collections import defaultdict, Counter

BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def load_data(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data

model_id = 2

models = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "deepseek-ai/deepseek-moe-16b-chat",
    "microsoft/Phi-3.5-MoE-instruct",
    "openai/gpt-oss-20b",
]

model_name = models[model_id].split("/")[-1]

n_experts = {
    "Qwen3-30B-A3B-Instruct-2507": 128,
    "deepseek-moe-16b-chat": 64,
    "Phi-3.5-MoE-instruct": 16,
    "gpt-oss-20b": 32,
}

n_expert = n_experts[model_name]

safety_classifications = load_data(BASE_FOLDER + f"precomputed_expert_analysis/{model_name}/{model_name}_safety_classifications_all_all.pkl")
sorted_experts_safe = defaultdict(list)
sorted_experts_unsafe = defaultdict(list)

for layer_name in safety_classifications:
    if layer_name.endswith("1"):
        counter = Counter(safety_classifications[layer_name])
        sorted_experts_unsafe[layer_name.replace("_1", "")] = counter.items()
    else:
        counter = Counter(safety_classifications[layer_name])
        sorted_experts_safe[layer_name.replace("_0", "")] = counter.items()

START_INDEX = 0
END_INDEX = n_expert-1 # Inclusive

fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=True)

axes_flat = axes.flatten()

for i, layer_name in enumerate(sorted_experts_unsafe):
    if i == 6:
        break
    
    ax = axes_flat[i]
    sorted_data_safe = sorted(sorted_experts_safe[layer_name], key=lambda item: item[0])
    sorted_data_unsafe = sorted(sorted_experts_unsafe[layer_name], key=lambda item: item[0])

    full_data_dict = {i: 0 for i in range(START_INDEX, END_INDEX + 1)}
    original_dict = dict(sorted_data_safe)
    full_data_dict.update(original_dict)
    filled_tuples_safe = list(full_data_dict.items())

    full_data_dict = {i: 0 for i in range(START_INDEX, END_INDEX + 1)}
    original_dict = dict(sorted_data_unsafe)
    full_data_dict.update(original_dict)
    filled_tuples_unsafe = list(full_data_dict.items())

    x_safe, y_safe = zip(*filled_tuples_safe)
    x_unsafe, y_unsafe = zip(*filled_tuples_unsafe)

    difference = tuple(b - a for a, b in zip(y_safe, y_unsafe))

    ax.plot(x_unsafe, y_unsafe, marker='o', linestyle='-', color='#FF4500', alpha=0.7, linewidth=1.5, label="unsafe")
    ax.plot(x_safe, y_safe, marker='o', linestyle='-', color='#007BFF', alpha=0.7, linewidth=1.5, label="safe")
    ax.plot(x_unsafe, difference, marker='o', linestyle='-', color='black', alpha=0.7, linewidth=1.5, label="difference")

    # plt.legend()
    ax.set_title(layer_name, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

fig.legend(
    ['safe', 'unsafe', 'difference'],
    loc='upper center',
    fontsize=10
)
plt.tight_layout()
plt.savefig(BASE_FOLDER + f'{model_name}_expert_choice.png')

# ----------------------------------------------------------------------------------------------------------------------------------

START_INDEX = 0
END_INDEX = n_expert-1 # Inclusive

fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=True)

axes_flat = axes.flatten()

for i, layer_name in enumerate(sorted_experts_unsafe):
    if i == 6:
        break
    
    ax = axes_flat[i]
    sorted_data_safe = sorted(sorted_experts_safe[layer_name], key=lambda item: item[0])
    sorted_data_unsafe = sorted(sorted_experts_unsafe[layer_name], key=lambda item: item[0])

    full_data_dict = {i: 0 for i in range(START_INDEX, END_INDEX + 1)}
    original_dict = dict(sorted_data_safe)
    full_data_dict.update(original_dict)
    filled_tuples_safe = list(full_data_dict.items())

    full_data_dict = {i: 0 for i in range(START_INDEX, END_INDEX + 1)}
    original_dict = dict(sorted_data_unsafe)
    full_data_dict.update(original_dict)
    filled_tuples_unsafe = list(full_data_dict.items())

    x_safe, y_safe = zip(*filled_tuples_safe)
    x_unsafe, y_unsafe = zip(*filled_tuples_unsafe)

    difference = tuple(b - a for a, b in zip(y_safe, y_unsafe))

    ax.plot(x_unsafe, y_unsafe, marker='o', linestyle='-', color='#FF4500', alpha=0.7, linewidth=1.5, label="unsafe")

    # plt.legend()
    ax.set_title(layer_name, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

fig.legend(
    ['unsafe'],
    loc='upper center',
    fontsize=10
)
plt.tight_layout()
plt.savefig(BASE_FOLDER + f'{model_name}_expert_choice_unsafe.png')