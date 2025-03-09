from ..default_models import ModelConfig, get_all_model_configs
from ..model_quality import QualityMetricsCollection, MMLU, MATH, GSM8K,  IFEval,  GPQA, Hellaswag, TLDR, TriviaQA, BIG_Bench

#### Phi3 Models ####

# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json
phi3mini_config = ModelConfig(model='microsoft/Phi-3-mini',
    hidden_size=3072, num_attention_heads=32, num_ffi = 2,
    intermediate_size=8192, num_decoder_layers=32,
    vocab_size=32064, max_model_len=128*1024, hidden_act="silu",
    sliding_window=256*1024,
    model_quality=QualityMetricsCollection([MMLU(accuracy=69.7, shots=5), BIG_Bench(accuracy=72.1, shots=3), Hellaswag(70.5, shots=5), GPQA(accuracy=29.7, shots=0), TriviaQA(57.8, shots=5), GSM8K(85.3, shots=8)]),
)
# https://huggingface.co/microsoft/Phi-3-small-128k-instruct/blob/main/config.json
phi3small_config = ModelConfig(model='microsoft/Phi-3-small',
    hidden_size=4096, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=8, head_dim=128,
    intermediate_size=14336, num_decoder_layers=32,
    vocab_size=100352, max_model_len=128*1024, hidden_act="gegelu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=75.5, shots=5), BIG_Bench(accuracy=77.6, shots=3), Hellaswag(79.6, shots=5), TriviaQA(66.0, shots=5), GSM8K(87.3, shots=8)]),
)

# https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json
phi3medium_config = ModelConfig(model='microsoft/Phi-3-medium',
    hidden_size=5120, num_attention_heads=40, num_ffi = 2,
    num_key_value_heads=10, head_dim=128,
    intermediate_size=17920, num_decoder_layers=40,
    vocab_size=32064, max_model_len=4*1024, hidden_act="silu",
    sliding_window=2047,
    model_quality=QualityMetricsCollection([MMLU(accuracy=76.6, shots=5), BIG_Bench(accuracy=77.9, shots=3), Hellaswag(81.6, shots=5), TriviaQA(73.9, shots=5), GSM8K(87.5, shots=8)]),
)

# https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/blob/main/config.json
phi3moe_config = ModelConfig(model='microsoft/Phi-3.5-MoE',
        hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=6400, num_decoder_layers=32,
    expert_top_k=2, num_experts=16,
    vocab_size=32064, max_model_len=128*1024, hidden_act="silu",
    sliding_window=128*1024,
    model_quality=QualityMetricsCollection([MMLU(accuracy=78.9, shots=5), BIG_Bench(accuracy=79.1, shots=0), GPQA(accuracy=36.8, shots=0) , Hellaswag(83.8, shots=5), MATH(59.5, shots=0), GSM8K(88.7, shots=8)]),
)

## Text embedding models
# https://huggingface.co/intfloat/multilingual-e5-small/blob/main/config.json
e5_small_config = ModelConfig(model='intfloat/multilingual-e5-small-instruct',
    hidden_size=384, num_attention_heads=12, num_ffi = 1,
    intermediate_size=1536, num_decoder_layers=12,
    vocab_size=250037, max_model_len=512, hidden_act="gelu",
)

# https://huggingface.co/intfloat/multilingual-e5-base/blob/main/config.json
e5_base_config = ModelConfig(model='intfloat/multilingual-e5-base-instruct',
    hidden_size=768, num_attention_heads=12, num_ffi = 1,
    intermediate_size=3072, num_decoder_layers=12,
    vocab_size=250002, max_model_len=514, hidden_act="gelu",
)

# https://huggingface.co/intfloat/multilingual-e5-large-instruct/blob/main/config.json
e5_large_config = ModelConfig(model='intfloat/multilingual-e5-large-instruct',
    hidden_size=1024, num_attention_heads=16, num_ffi = 1,
    intermediate_size=4096, num_decoder_layers=24,
    vocab_size=250002, max_model_len=514, hidden_act="gelu",
)

microsoft_models = get_all_model_configs(__name__)
microsoft_models.update({
    'microsoft/phi3medium': phi3medium_config,
    'microsoft/phi3mini': phi3mini_config,
    'microsoft/phi3small': phi3small_config,
    'microsoft/phi3moe': phi3moe_config,
    'phi3medium': phi3medium_config,
    'phi3mini': phi3mini_config,
    'phi3small': phi3small_config,
    'e5_small': e5_small_config,
    'e5_base': e5_base_config,
    'e5_large': e5_large_config,
})
