import matplotlib.pyplot as plt
import pickle
from collections import defaultdict, Counter

def load_dict(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data

models = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
model_name = models[0].split("/")[-1]

# 1. Define your dictionary
safety_classifications = load_dict(f"../pre_computed_act/safety_classifications_{model_name}.p")

sorted_experts_safe = defaultdict(list)
sorted_experts_unsafe = defaultdict(list)
for layer_name in safety_classifications:
    counter = Counter(safety_classifications[layer_name])
    if layer_name.endswith("1"):
        sorted_experts_unsafe[layer_name.replace("gate_1", "gate")] = counter.items()
    else:
        sorted_experts_safe[layer_name.replace("gate_0", "gate")] = counter.items()

START_INDEX = 0
END_INDEX = 127 # Inclusive

fig, axes = plt.subplots(7, 7, figsize=(110, 75), sharex=True, sharey=True)

axes_flat = axes.flatten()

for i, layer_name in enumerate(sorted_experts_unsafe):
    ax = axes_flat[i]
    sorted_data_safe = sorted(sorted_experts_safe[layer_name], key=lambda item: item[0])
    sorted_data_unsafe = sorted(sorted_experts_unsafe[layer_name], key=lambda item: item[0])

    print(sorted_data_safe)
    print(sorted_data_unsafe)
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
    print(difference)

    ax.plot(x_safe, y_safe, marker='o', linestyle='-', color='#007BFF', alpha=0.7, linewidth=1.5, label="safe")
    # plt.plot(x_unsafe, [-y for y in y_unsafe], marker='o', linestyle='-', color='#FF4500', alpha=0.7, linewidth=1.5)
    ax.plot(x_unsafe, y_unsafe, marker='o', linestyle='-', color='#FF4500', alpha=0.7, linewidth=1.5, label="unsafe")
    ax.plot(x_unsafe, difference, marker='o', linestyle='-', color='black', alpha=0.7, linewidth=1.5, label="differnce")

    # plt.legend()
    ax.set_title(layer_name, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

fig.legend(
    ax.get_lines(), # Use lines from the last axis
    ['safe', 'unsafe', 'difference'],
    loc='upper center',
    bbox_to_anchor=(0.99, 0.99),
    fontsize=10
)

# fig.xlabel('Expert', fontsize=12)
# fig.ylabel('Token count', fontsize=12)
plt.tight_layout()
plt.savefig('expert_choice.png')
