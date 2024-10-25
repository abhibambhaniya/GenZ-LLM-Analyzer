from ..default_models import ModelConfig, get_all_model_configs


# https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/config.json
falcon7b_config = ModelConfig(model='tiiuae/falcon-7b-instruct',
    hidden_size=4544, num_attention_heads=71, num_ffi = 1,
    num_key_value_heads=71, head_dim=64,
    intermediate_size=4544*4, num_decoder_layers=32,
    vocab_size=65024, max_model_len=2*1024, hidden_act="relu",
)

# https://huggingface.co/databricks/dbrx-base/blob/main/config.json
dbrx_config = ModelConfig(model='databricks/dbrx-base',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=10752, num_decoder_layers=40,
    expert_top_k=4, num_experts=16, moe_layer_freq=1,
    max_model_len=32*1024, vocab_size=100352, hidden_act="silu",
)

gpt_4_config = ModelConfig(model='openai/GPT-4',
    hidden_size=84*128, num_attention_heads=84,
    num_key_value_heads=84, num_ffi = 1,
    intermediate_size=4*84*128, num_decoder_layers=128,
    expert_top_k=2, num_experts=16, moe_layer_freq=1,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
)

# https://huggingface.co/xai-org/grok-1/blob/main/RELEASE
grok_1_config = ModelConfig(model='xai-org/grok-1',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 1,
    intermediate_size=8*6144, num_decoder_layers=64,
    expert_top_k=2, num_experts=8, moe_layer_freq=1,
    vocab_size=128*1024, max_model_len=8*1024, hidden_act="gelu",
)

# https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/config.json
glm_9b_config = ModelConfig(model='THUDM/glm-4-9b-chat',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=2, num_ffi = 2,
    intermediate_size=13696, num_decoder_layers=40,
    max_model_len=131072, vocab_size=151552, hidden_act="gelu",
)

# https://huggingface.co/state-spaces/mamba-130m-hf/blob/main/config.json
mamba_130m_config = ModelConfig(model='state-spaces/mamba-130m-hf',
    hidden_size=768, num_attention_heads=1,
    num_key_value_heads=1, num_ffi = 2,
    intermediate_size=1536, num_decoder_layers=24,
    max_model_len=131072, vocab_size=50280, hidden_act="silu",
    mamba_d_state = 16, mamba_dt_rank = 48, mamba_expand = 2, mamba_d_conv=4,
)

# https://huggingface.co/state-spaces/mamba-2.8b-hf/blob/main/config.json
mamba_3b_config = ModelConfig(model='state-spaces/mamba-2.8b-hf',
    hidden_size=2560, num_attention_heads=1,
    num_key_value_heads=1, num_ffi = 2,
    intermediate_size=5120, num_decoder_layers=64,
    max_model_len=131072, vocab_size=50280, hidden_act="silu",
    mamba_d_state = 16, mamba_dt_rank = 160, mamba_expand = 2, mamba_d_conv=4,
)

# https://huggingface.co/tiiuae/falcon-mamba-7b/blob/main/config.json
falcon_mamba_7b_config = ModelConfig(model='tiiuae/falcon-mamba-7b',
    hidden_size=4096, num_attention_heads=1,
    num_key_value_heads=1, num_ffi = 2,
    intermediate_size=8192, num_decoder_layers=64,
    max_model_len=131072, vocab_size=50280, hidden_act="silu",
    mamba_d_state = 16, mamba_dt_rank = 256, mamba_expand = 16, mamba_d_conv=4,
)


super_llm_config = ModelConfig(model='SuperLLM-10T',
    hidden_size=108*128, num_attention_heads=108,
    num_key_value_heads=108, num_ffi = 2,
    intermediate_size=4*108*128, num_decoder_layers=128,
    expert_top_k=4, num_experts=32, moe_layer_freq=1
)



# https://huggingface.co/HuggingFaceH4/zephyr-7b-beta/blob/main/config.json
zephyr_7b_config = ModelConfig(model='HuggingFaceH4/zephyr-7b-beta',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
    vocab_size=32000, max_model_len=32*1024, sliding_window=4096,
    hidden_act="silu")

# https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/config.json
deep_seek_moe_16b_config = ModelConfig(model='deepseek-ai/deepseek-moe-16b-base',
    hidden_size=2048, num_attention_heads=16,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=10944, num_decoder_layers=28,
    expert_top_k=6, num_experts=64, moe_layer_freq=1,
    moe_intermediate_size=1408,
    n_shared_experts=2, shared_expert_intermediate_size=1408,
    first_k_dense_replace = 1,
    vocab_size=102400, max_model_len=4*1024,  hidden_act="silu",
)
## TODO: account for shared expert, shared account is regular MLP which is always added.
## This has a special case where the first layer is dense and the rest are MoE with shared experts.
## MLP in this case is n_shared_experts*shared_expert_intermediate_size + Activated*moe_intermediate_size

misc_models = get_all_model_configs()

misc_models.update({
    'glm-9b': glm_9b_config,
    'HuggingFaceH4/zephyr-7b-beta': zephyr_7b_config,
    'THUDM/glm-4-9b-chat': glm_9b_config,
    'databricks/dbrx-base': dbrx_config,
    'dbrx': dbrx_config,
    'falcon7b': falcon7b_config,
    'gpt-4': gpt_4_config,
    'grok-1': grok_1_config,
    'openai/gpt-4': gpt_4_config,
    'state-spaces/mamba-130m-hf': mamba_130m_config,
    'state-spaces/mamba-2.8b-hf': mamba_3b_config,
    'nvidia/mamba2-8b-3t-4k': mamba_3b_config,
    'super_llm': super_llm_config,
    'tiiuae/falcon-7b-instruct': falcon7b_config,
    'tiiuae/falcon-mamba-7b': falcon_mamba_7b_config,
    'xai-org/grok-1': grok_1_config,
})