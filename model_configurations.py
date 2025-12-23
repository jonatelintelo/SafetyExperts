from dataclasses import dataclass
from typing import Optional

@dataclass
class MoEModelConfig:
    model_name: str
    gate_name: str
    top_k: int
    num_router_expert: int
    name_router_expert: str
    name_expert_layers: str
    name_shared_expert_gate: Optional[str] = 'None'
    name_shared_expert: Optional[str] = 'None'

models = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": MoEModelConfig(
        model_name="Qwen3-30B-A3B-Instruct-2507", 
        gate_name="mlp.gate", 
        top_k=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": MoEModelConfig(
        model_name="Qwen3-30B-A3B-Thinking-2507", 
        gate_name="mlp.gate", 
        top_k=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "Qwen/Qwen3-30B-A3B": MoEModelConfig(
        model_name="Qwen3-30B-A3B", 
        gate_name="mlp.gate", 
        top_k=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "microsoft/Phi-3.5-MoE-instruct": MoEModelConfig(
        model_name="Phi-3.5-MoE-instruct", 
        gate_name="block_sparse_moe.gate", 
        top_k=2,
        num_router_expert=16,
        name_router_expert="block_sparse_moe.experts",
        name_expert_layers=["w1", "w3"]),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": MoEModelConfig(
        model_name="Mixtral-8x7B-Instruct-v0.1", 
        gate_name="block_sparse_moe.gate", 
        top_k=2, 
        num_router_expert=8,
        name_router_expert="block_sparse_moe.experts",
        name_expert_layers=["w1", "w3"]),
    "openai/gpt-oss-20b": MoEModelConfig(
        model_name="gpt-oss-20b", 
        gate_name="mlp.router", 
        top_k=4, 
        num_router_expert=32,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_up_layers"]),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": MoEModelConfig(
        model_name="Qwen1.5-MoE-A2.7B-Chat", 
        gate_name="mlp.gate", # and mlp.shared_expert_gate for the (4) shared experts
        top_k=4, 
        num_router_expert=60,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert_gate="mlp.shared_expert_gate",
        name_shared_expert="mlp.shared_expert"),
    "tencent/Hunyuan-A13B-Instruct": MoEModelConfig(
        model_name="Hunyuan-A13B-Instruct", 
        gate_name="mlp.gate.wg",
        top_k=8, 
        num_router_expert=64,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_mlp"),
    "deepseek-ai/deepseek-moe-16b-chat": MoEModelConfig(
        model_name="deepseek-moe-16b-chat", 
        gate_name="mlp.gate", 
        top_k=6, 
        num_router_expert=64,
        name_router_expert="mlp.experts", # the first layer only have shared expert
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_experts"),
    "IntervitensInc/pangu-pro-moe-model": MoEModelConfig(
        model_name="pangu-pro-moe-model", 
        gate_name="mlp.gate", 
        top_k=8, 
        num_router_expert=64,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_expert"),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": MoEModelConfig(
        model_name="Llama-4-Scout-17B-16E-Instruct", 
        gate_name="feed_forward.router", 
        top_k=1, 
        num_router_expert=16,
        name_router_expert="feed_forward.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="feed_forward.shared_expert"),
    "moonshotai/Kimi-VL-A3B-Instruct": MoEModelConfig(
        model_name="Kimi-VL-A3B-Instruct", 
        gate_name="mlp.gate", 
        top_k=6, 
        num_router_expert=64,
        name_router_expert="mlp.experts", # the first layer only have shared expert
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_experts"),
    "moonshotai/Kimi-VL-A3B-Thinking": MoEModelConfig(
        model_name="Kimi-VL-A3B-Thinking", 
        gate_name="mlp.gate", 
        top_k=6, 
        num_router_expert=64,
        name_router_expert="mlp.experts", # the first layer only have shared expert
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_experts"),
    "moonshotai/Kimi-VL-A3B-Thinking-2506": MoEModelConfig(
        model_name="Kimi-VL-A3B-Thinking-2506", 
        gate_name="mlp.gate", 
        top_k=6, 
        num_router_expert=64,
        name_router_expert="mlp.experts", # the first layer only have shared expert
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_experts"),
}
