from ..default_models import ModelConfig, get_all_model_configs


#### Phi3 Models ####

# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json
phi3mini_config = ModelConfig(model='microsoft/Phi-3-mini',
    hidden_size=3072, num_attention_heads=32, num_ffi = 2,
    intermediate_size=8192, num_decoder_layers=32,
    vocab_size=32064, max_model_len=128*1024, hidden_act="silu",
    sliding_window=256*1024
)
# https://huggingface.co/microsoft/Phi-3-small-128k-instruct/blob/main/config.json
phi3small_config = ModelConfig(model='microsoft/Phi-3-small',
    hidden_size=4096, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=8, head_dim=128,
    intermediate_size=14336, num_decoder_layers=32,
    vocab_size=100352, max_model_len=128*1024, hidden_act="gegelu",
    sliding_window=256*1024
)

# https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json
phi3medium_config = ModelConfig(model='microsoft/Phi-3-medium',
    hidden_size=5120, num_attention_heads=40, num_ffi = 2,
    num_key_value_heads=10, head_dim=128,
    intermediate_size=17920, num_decoder_layers=40,
    vocab_size=32064, max_model_len=4*1024, hidden_act="silu",
    sliding_window=2047
)

# https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/blob/main/config.json
phi3moe_config = ModelConfig(model='microsoft/Phi-3.5-MoE',
        hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=6400, num_decoder_layers=32,
    expert_top_k=2, num_experts=16, moe_layer_freq=1,
    vocab_size=32064, max_model_len=128*1024, hidden_act="silu",
    sliding_window=128*1024
)

microsoft_models = get_all_model_configs()
microsoft_models.update({
    'microsoft/phi3medium': phi3medium_config,
    'microsoft/phi3mini': phi3mini_config,
    'microsoft/phi3small': phi3small_config,
    'microsoft/phi3moe': phi3moe_config,
    'phi3medium': phi3medium_config,
    'phi3mini': phi3mini_config,
    'phi3small': phi3small_config,
})
