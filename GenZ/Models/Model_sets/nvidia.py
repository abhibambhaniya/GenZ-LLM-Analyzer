from ..default_models import ModelConfig, get_all_model_configs

##### Nvidia Models ########
# https://huggingface.co/nvidia/Nemotron-4-340B-Instruct/blob/main/model_config.yaml
nemotron_340b_config = ModelConfig(model='nvidia/Nemotron-4-340B-Instruct',
    hidden_size=18432, num_attention_heads=96,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=73728, num_decoder_layers=96,
    vocab_size=256000, max_model_len=4*1024, hidden_act="silu",
)

# TODO: Addition changes required in Genz as has different layerwise configs
# https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct/blob/main/config.json
# nemotron_51b_config = ModelConfig(model='nvidia/Llama-3_1-Nemotron-51B-Instruct',

#     hidden_size=8192, num_attention_heads=64,
#     num_key_value_heads=8, num_ffi = 2,
#     intermediate_size=28672, num_decoder_layers=80,
#     vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
# )


# https://arxiv.org/pdf/2402.16819
nemotron_15b_config = ModelConfig(model='nvidia/Nemotron-4-15B',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=4*6144, num_decoder_layers=32,
    vocab_size=256000, max_model_len=4*1024, hidden_act="relu",
)

# https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base/blob/main/config.json
minitron_8b_config = ModelConfig(model='nvidia/Mistral-NeMo-Minitron-8B-Base',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=11520, num_decoder_layers=40,
    vocab_size=131072, max_model_len=8*1024, hidden_act="silu",
)

nvidia_models = get_all_model_configs(__name__)
nvidia_models.update({
    # 'nvidia/Llama-3_1-Nemotron-51B-Instruct': nemotron_51b_config,
    'nvidia/Mistral-NeMo-Minitron-8B-Base': minitron_8b_config,
    'nvidia/Nemotron-4-15B': nemotron_15b_config,
    'nvidia/Nemotron-4-340B-Instruct': nemotron_340b_config,

})
