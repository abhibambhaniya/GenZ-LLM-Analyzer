from ..default_models import ModelConfig, get_all_model_configs

##### Qwen1 Models ########
# https://huggingface.co/Qwen/Qwen-1_8B/blob/main/config.json
qwen1_8b_config = ModelConfig(model='Qwen/Qwen-1_8B',
    hidden_size=2048, num_attention_heads=16,
    num_key_value_heads=16, num_ffi = 1,
    intermediate_size=11008, num_decoder_layers=24,
    vocab_size=151936, max_model_len=8*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen-7B/blob/main/config.json
qwen1_7b_config = ModelConfig(model='Qwen/Qwen-7B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=32, num_ffi = 1,
    intermediate_size=22016, num_decoder_layers=32,
    vocab_size=151936, max_model_len=8*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen-14B/blob/main/config.json
qwen1_14b_config = ModelConfig(model='Qwen/Qwen-14B',
    hidden_size=5120, num_attention_heads=40,
    num_key_value_heads=40, num_ffi = 1,
    intermediate_size=27392, num_decoder_layers=40,
    vocab_size=152064, max_model_len=2*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen-72B/blob/main/config.json
qwen1_72b_config = ModelConfig(model='Qwen/Qwen-72B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=40, num_ffi = 1,
    intermediate_size=49152, num_decoder_layers=80,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

##### Qwen1.5 Models ########
# https://huggingface.co/Qwen/Qwen1.5-0.5B/blob/main/config.json
qwen1_5_0_5b_config = ModelConfig(model='Qwen/Qwen1.5-0.5B',
    hidden_size=1024, num_attention_heads=16,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=2816, num_decoder_layers=24,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-1.8B/blob/main/config.json
qwen1_5_1_8b_config = ModelConfig(model='Qwen/Qwen1.5-1.8B',
    hidden_size=2048, num_attention_heads=16,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=5504, num_decoder_layers=24,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-4B/blob/main/config.json
qwen1_5_4b_config = ModelConfig(model='Qwen/Qwen1.5-4B',
    hidden_size=2560, num_attention_heads=20,
    num_key_value_heads=20, num_ffi = 2,
    intermediate_size=6912, num_decoder_layers=40,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-7B/blob/main/config.json
qwen1_5_7b_config = ModelConfig(model='Qwen/Qwen1.5-7B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=32, num_ffi = 2,
    intermediate_size=11008, num_decoder_layers=32,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json
qwen1_5_14b_config = ModelConfig(model='Qwen/Qwen1.5-14B',
    hidden_size=5120, num_attention_heads=40,
    num_key_value_heads=40, num_ffi = 2,
    intermediate_size=13696, num_decoder_layers=40,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/config.json
qwen1_5_32b_config = ModelConfig(model='Qwen/Qwen1.5-32B',
    hidden_size=5120, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=27392, num_decoder_layers=64,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-72B/blob/main/config.json
qwen1_5_72b_config = ModelConfig(model='Qwen/Qwen1.5-72B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=64, num_ffi = 2,
    intermediate_size=24576, num_decoder_layers=80,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-110B/blob/main/config.json
qwen1_5_110b_config = ModelConfig(model='Qwen/Qwen1.5-110B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=64, num_ffi = 2,
    intermediate_size=49152, num_decoder_layers=80,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json
qwen1_5_moe_2_7b_config = ModelConfig(model='Qwen/Qwen1.5-MoE-A2.7B',
    hidden_size=2048, num_attention_heads=16,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=5632, num_decoder_layers=40,
    expert_top_k=4, num_experts=60, moe_layer_freq=1,
    moe_intermediate_size=1408, shared_expert_intermediate_size=5632,
    vocab_size=151936, max_model_len=8*1024, sliding_window=32*1024, hidden_act="silu",
)
## TODO: account for shared expert, shared account is regular MLP which is always added.
## MLP in this case is shared_expert_intermediate_size + Activated*moe_intermediate_size

##### Qwen2 Models ########
# https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json
qwen2_0_5b_config = ModelConfig(model='Qwen/Qwen2-0.5B',
    hidden_size=896, num_attention_heads=14,
    num_key_value_heads=2, num_ffi = 2,
    intermediate_size=4864, num_decoder_layers=24,
    vocab_size=151936, max_model_len=128*1024, sliding_window=128*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
qwen2_1_5b_config = ModelConfig(model='Qwen/Qwen2.5-1.5B',
    hidden_size=1536, num_attention_heads=12,
    num_key_value_heads=2, num_ffi = 2,
    intermediate_size=8960, num_decoder_layers=28,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2-3B-Instruct/blob/main/config.json
qwen2_3b_config = ModelConfig(model='Qwen/Qwen2-3B',
    hidden_size=2048, num_attention_heads=16,
    num_key_value_heads=2, num_ffi = 2,
    intermediate_size=11008, num_decoder_layers=36,
    vocab_size=151936, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json
qwen2_7b_config = ModelConfig(model='Qwen/Qwen2-7B',
    hidden_size=3584, num_attention_heads=28,
    num_key_value_heads=4, num_ffi = 2,
    intermediate_size=18944, num_decoder_layers=28,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2-14B-Instruct/blob/main/config.json
qwen2_14b_config = ModelConfig(model='Qwen/Qwen2-14B',
    hidden_size=5120, num_attention_heads=40,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=13824, num_decoder_layers=48,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2-32B-Instruct/blob/main/config.json
qwen2_32b_config = ModelConfig(model='Qwen/Qwen2-32B',
    hidden_size=5120, num_attention_heads=40,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=27648, num_decoder_layers=64,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")

# https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/config.json
qwen2_72b_config = ModelConfig(model='Qwen/Qwen2-72B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=29568, num_decoder_layers=80,
    vocab_size=152064, max_model_len=32*1024, sliding_window=32*1024,hidden_act="silu")


alibaba_models = get_all_model_configs()

