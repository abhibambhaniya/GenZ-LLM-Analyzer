from ..default_models import ModelConfig, get_all_model_configs

#### OPT Models ####

# https://huggingface.co/openai-community/gpt2/blob/main/config.json
gpt2_config = ModelConfig(model='openai/gpt2',
    hidden_size=768, num_attention_heads=12, num_ffi = 1,
    intermediate_size=4*768, num_decoder_layers=12,
    vocab_size=50257, hidden_act="gelu_new", max_model_len=1024
)

# https://huggingface.co/facebook/opt-125m/blob/main/config.json
opt_125m_config = ModelConfig(model='facebook/opt-125M',
    hidden_size=768, num_attention_heads=12, num_ffi = 1,
    intermediate_size=4*768, num_decoder_layers=12,
    vocab_size=50272, hidden_act="relu", max_model_len=2*1024
)

# https://huggingface.co/facebook/opt-350m/blob/main/config.json
opt_350m_config = ModelConfig(model='facebook/OPT-350M',
    hidden_size=1024, num_attention_heads=16, num_ffi = 1,
    intermediate_size=4*1024, num_decoder_layers=24,
    vocab_size=50272, hidden_act="relu", max_model_len=2*1024
)

# https://huggingface.co/facebook/opt-1.3b/blob/main/config.json
opt_1b_config = ModelConfig(model='facebook/OPT-1.3b',
    hidden_size=2048, num_attention_heads=32, num_ffi = 1,
    intermediate_size=4*2048, num_decoder_layers=24,
    max_model_len=2*1024, vocab_size=50272, hidden_act="relu",
)

# https://huggingface.co/facebook/opt-6.7b/blob/main/config.json
opt_7b_config = ModelConfig(model='facebook/opt-6.7b',
    hidden_size=4096, num_attention_heads=32, num_ffi = 1,
    intermediate_size=4*4096, num_decoder_layers=32,
    max_model_len=2*1024, vocab_size=50272, hidden_act="relu",
)

# https://huggingface.co/facebook/opt-13b/blob/main/config.json
opt_13b_config = ModelConfig(model='facebook/OPT-13B',
    hidden_size=5140, num_attention_heads=40, num_ffi = 1,
    intermediate_size=4*5140, num_decoder_layers=40,
    max_model_len=2*1024, vocab_size=50272, hidden_act="relu",
)

# https://huggingface.co/facebook/opt-30b/blob/main/config.json
opt_30b_config = ModelConfig(model='facebook/opt-30B',
    hidden_size=7168, num_attention_heads=56,
    intermediate_size=4*7168, num_decoder_layers=48,
    max_model_len=2*1024, vocab_size=50272, hidden_act="relu",
)

# https://huggingface.co/intlsy/opt-175b-hyperparam/blob/main/config.json
gpt3_config = ModelConfig(model='openai/GPT3-175B',
    hidden_size=12288, num_attention_heads=96, num_ffi = 1,
    intermediate_size=4*12288, num_decoder_layers=96,
    max_model_len=2*1024, vocab_size=50272, hidden_act="relu",
)

######## LLaMA Models ########

# https://huggingface.co/Secbone/llama-33B-instructed/blob/main/config.json
llama_33b_config = ModelConfig(model='meta-llama/Llama-33B',
    hidden_size=6656, num_attention_heads=52, num_ffi = 2,
    intermediate_size=17920, num_decoder_layers=60,
    vocab_size=32000, max_model_len=2*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
llama2_7b_config = ModelConfig(model='meta-llama/Llama-2-7B',
    hidden_size=4096, num_attention_heads=32, num_ffi = 2,
    intermediate_size=11008, num_decoder_layers=32,
    vocab_size=32000, max_model_len=4*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
llama_13b_config = ModelConfig(model='meta-llama/Llama-2-13B',
    hidden_size=5120, num_attention_heads=40, num_ffi = 2,
    intermediate_size=13824, num_decoder_layers=40,
    vocab_size=32000, max_model_len=4*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
llama_70b_config = ModelConfig(model='meta-llama/Llama-2-70B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=28672, num_decoder_layers=80,
    vocab_size=32000, max_model_len=4*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
llama3_8b_config = ModelConfig(model='meta-llama/Llama-3.1-8B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",

)

# https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/config.json
llama3_70b_config = ModelConfig(model='meta-llama/Llama-3.1-70B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=28672, num_decoder_layers=80,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Meta-Llama-3.1-405B
llama3_405b_config = ModelConfig(model='meta-llama/Llama-3.1-405B',
    hidden_size=16384, num_attention_heads=128,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=3.25*16384, num_decoder_layers=126,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",

)

# https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
llama3_2_1b_config = ModelConfig(model='meta-llama/Llama-3.2-1B',
    hidden_size=2048, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=4*2048, num_decoder_layers=16,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
)

# https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
llama3_2_3b_config = ModelConfig(model='meta-llama/Llama-3.2-3B',
    hidden_size=3072, num_attention_heads=24,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=4*2048, num_decoder_layers=28,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
)

meta_models = get_all_model_configs(__name__)
meta_models.update({
    'facebook/opt-1.3b': opt_1b_config,
    'facebook/opt-125m': opt_125m_config,
    'facebook/opt-13b': opt_13b_config,
    'facebook/opt-175b': gpt3_config,
    'facebook/opt-30B': opt_30b_config,
    'facebook/opt-350m': opt_350m_config,
    'facebook/opt-7b': opt_7b_config,
    'gpt-3_1b': opt_1b_config,
    'gpt-2': gpt2_config,
    'gpt-3_7b': opt_7b_config,
    'opt_125m': opt_125m_config,
    'opt_13b': opt_13b_config,
    'opt_175b': gpt3_config,
    'opt_1b': opt_1b_config,
    'opt_30b': opt_30b_config,
    'opt_350m': opt_350m_config,
    'opt_7b': opt_7b_config,
    'llama_33b': llama_33b_config,
    'llama2_7b': llama2_7b_config,
    'llama2_13b': llama_13b_config,
    'llama2_70b': llama_70b_config,
    'llama3_8b': llama3_8b_config,
    'llama_405b': llama3_405b_config,
    'meta-llama/Llama-3.2-1B': llama3_2_1b_config,
    'meta-llama/Llama-3.2-3B': llama3_2_3b_config,
    'meta-llama/llama-2-13b': llama_13b_config,
    'meta-llama/llama-2-33b': llama_33b_config,
    'meta-llama/llama-2-70b': llama_70b_config,
    'meta-llama/llama-2-7b': llama2_7b_config,
    'meta-llama/meta-llama-3.1-405b': llama3_405b_config,
    'meta-llama/meta-llama-3.1-70b': llama_70b_config,
    'meta-llama/meta-llama-3.1-8b': llama3_8b_config,
    'gpt-3': gpt3_config,
    'openai/gpt-3': gpt3_config,
})