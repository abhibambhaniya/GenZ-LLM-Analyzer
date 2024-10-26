from ..default_models import ModelConfig, get_all_model_configs
##### Gemma Models #####
# https://huggingface.co/google/gemma-2b-it/blob/main/config.json
gemma_2b_config = ModelConfig(model='google/gemma-2B',
    hidden_size=2048, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=1, head_dim=256,
    intermediate_size=16384, num_decoder_layers=18,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu",
)

# https://huggingface.co/google/gemma-7b-it/blob/main/config.json
gemma_7b_config = ModelConfig(model='google/gemma-7B',
    hidden_size=3072, num_attention_heads=16, num_ffi = 2,
    intermediate_size=24576, num_decoder_layers=28, head_dim=256,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu",
)

# https://huggingface.co/google/gemma-2-2b/blob/main/config.json
gemma2_2b_config = ModelConfig(model='google/gemma-2-2B',
    hidden_size=2304, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=4, head_dim=256,
    intermediate_size=9216, num_decoder_layers=26,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
)

# https://huggingface.co/google/gemma-2-9b/blob/main/config.json
gemma2_9b_config = ModelConfig(model='google/gemma-2-9B',
    hidden_size=3584, num_attention_heads=16, num_ffi = 2,
    num_key_value_heads=8, head_dim=256,
    intermediate_size=14336, num_decoder_layers=42,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
)

# https://huggingface.co/google/gemma-2-27b-it/blob/main/config.json
gemma2_27b_config = ModelConfig(model='google/gemma-2-27B',
    hidden_size=4608, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=16, head_dim=128,
    intermediate_size=36864, num_decoder_layers=46,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
)






####################
palm_config = ModelConfig(model='google/palm',
        hidden_size=18432, num_attention_heads=48, num_ffi = 1,
        intermediate_size=4*18432, num_decoder_layers=118
)

google_models = get_all_model_configs(__name__)

google_models.update({
    'gemma2_27b': gemma2_27b_config,
    'gemma2_9b': gemma2_9b_config,
    'gemma2_2b': gemma2_2b_config,
    'gemma_2b': gemma_2b_config,
    'gemma_7b': gemma_7b_config,
    'google/gemma-2-27b-it': gemma2_27b_config,
    'google/gemma-2-9b': gemma2_9b_config,
    'google/gemma-2-2b': gemma2_2b_config,
    'google/gemma-2b-it': gemma_2b_config,
    'google/gemma-7b-it': gemma_7b_config,
    'google/palm': palm_config,
    'palm': palm_config,
})