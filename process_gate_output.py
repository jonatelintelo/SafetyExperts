import model_configurations
import data_utils


BASE_FOLDER = "/home/jtelintelo/SafetyExperts/results/"

def get_token_traces(activations_dict):
    """
    Reshapes layer-wise MoE data into token-wise traces.
    
    Args:
        activations_dict: Dictionary where keys are layer names 
                          and values are lists of tensors (one tensor per prompt).
    
    Returns:
        traces: dict where traces[prompt_index][token_index] = list of 32 layers' experts
                e.g., traces[0][5] might look like [[1, 4], [3, 7], ... ]
    """
    traces = {}

    # 1. Sort the layer keys numerically (0 to 31) 
    # This ensures we build the path in the correct order of depth.
    layer_keys = sorted(
        activations_dict.keys(), 
        key=lambda x: int(x.split('.')[2])  # Extracts '0' from 'model.layers.0...'
    )

    # print(f"layer_keys: '{layer_keys}'")

    # 2. Determine number of prompts by looking at Layer 0
    # We assume the list of tensors is the same length across all layers
    first_layer_data = activations_dict[layer_keys[0]]
    # print(f"first_layer_data: '{first_layer_data}'")

    num_prompts = len(first_layer_data)
    # print(f"num_prompts: '{num_prompts}'")

    for prompt_idx in range(num_prompts):
        traces[prompt_idx] = {}
        
        # 3. Determine number of tokens for THIS specific prompt
        # We look at the tensor shape in the first layer for this prompt
        # e.g., torch.Size([18, 2]) -> num_tokens = 18
        tensor_shape = first_layer_data[prompt_idx].shape
        num_tokens = tensor_shape[0]
        # print(f"num_tokens: '{num_tokens}'")

        # Initialize the list for every token in this prompt
        for token_idx in range(num_tokens):
            traces[prompt_idx][token_idx] = []

        # 4. Iterate through every layer to build the trace
        for layer_key in layer_keys:
            # Get the tensor for the current prompt in the current layer
            # Shape is [num_tokens, 2]
            layer_tensor = activations_dict[layer_key][prompt_idx]
            
            # Extract the experts for each token
            for token_idx in range(num_tokens):
                # .tolist() converts the tensor row [ExpA, ExpB] into a python list
                experts = layer_tensor[token_idx].tolist()
                traces[prompt_idx][token_idx].append(experts)

    return traces

if __name__ == "__main__":
    model_id = 0

    models = [
        # LLMs
        "Qwen/Qwen3-30B-A3B-Instruct-2507",     #0
        "microsoft/Phi-3.5-MoE-instruct",       #1
        "mistralai/Mixtral-8x7B-Instruct-v0.1", #2
        "openai/gpt-oss-20b",                   #3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",          #4
        "tencent/Hunyuan-A13B-Instruct",        #5
        "deepseek-ai/deepseek-moe-16b-chat",    #6
        "IntervitensInc/pangu-pro-moe-model",   #7
        # VLMs
        "moonshotai/Kimi-VL-A3B-Instruct",      #8
        "moonshotai/Kimi-VL-A3B-Thinking",      #9
        "moonshotai/Kimi-VL-A3B-Thinking-2506", #10
        "deepseek-ai/deepseek-vl2-small",       #11
        "deepseek-ai/deepseek-vl2",             #12
    ]
    
    model_config = model_configurations.models[models[model_id]]

    print(f"Tested Model: '{model_config.model_name}'")
    
    print("\nLoading precomputed activation analysis...")
    expert_indices = data_utils.load_data(BASE_FOLDER + f"gate_output/{model_config.model_name}/{model_config.model_name}_expert_indices.pkl")

    # for layer_name in expert_indices:
    #     print(f"Number of prompts recorded in layer '{layer_name}': '{len(expert_indices[layer_name])}'")

    #     print(expert_indices[layer_name][0])
    #     print(expert_indices[layer_name][14915])
        
    #     for prompt_index, prompt_expert_indices in enumerate(expert_indices[layer_name]):
    #         print(f"Layer: '{layer_name}', prompt_index: '{prompt_index}', shape: '{prompt_expert_indices.shape}'")

    # Run the reshaping
    traces = get_token_traces(expert_indices)
    # _, labels = data_utils.load_datasets(malicious_only=False) # label 1 = refusal behavior
    _, labels = data_utils.load_twinset()
    
    print(labels[0])
    print(traces[0])

    data_utils.create_directory(BASE_FOLDER + f"{model_config.model_name}/data/")

    data_utils.save_data(traces, BASE_FOLDER + f"{model_config.model_name}/data/{model_config.model_name}_input.pkl")
    data_utils.save_data(labels, BASE_FOLDER + f"{model_config.model_name}/data/{model_config.model_name}_labels.pkl")

    print("\n------------------ Job Finished ------------------")
