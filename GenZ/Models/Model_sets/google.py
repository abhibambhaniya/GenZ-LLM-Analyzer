from ..default_models import ModelConfig, get_all_model_configs
from ..model_quality import QualityMetricsCollection, MMLU, MATH, GSM8K,  IFEval,  GPQA, Hellaswag, TLDR, TriviaQA, BIG_Bench
##### Gemma Models #####
# https://huggingface.co/google/gemma-2b-it/blob/main/config.json
gemma_2b_config = ModelConfig(model='google/gemma-2B',
    hidden_size=2048, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=1, head_dim=256,
    intermediate_size=16384, num_decoder_layers=18,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=42.3, shots=5), Hellaswag(accuracy=71.4, shots=0), MATH(accuracy=11.8, shots=4), GSM8K(accuracy=17.7), TriviaQA(accuracy=53.2, shots=5), BIG_Bench(accuracy=35.2)]),
)

# https://huggingface.co/google/gemma-7b-it/blob/main/config.json
gemma_7b_config = ModelConfig(model='google/gemma-7B',
    hidden_size=3072, num_attention_heads=16, num_ffi = 2,
    intermediate_size=24576, num_decoder_layers=28, head_dim=256,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=64.3, shots=5), Hellaswag(accuracy=81.2, shots=0), MATH(accuracy=24.3, shots=4), GSM8K(accuracy=46.4), TriviaQA(accuracy=63.4, shots=5), BIG_Bench(accuracy=55.1)]),
)

# https://huggingface.co/google/gemma-2-2b/blob/main/config.json
gemma2_2b_config = ModelConfig(model='google/gemma-2-2B',
    hidden_size=2304, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=4, head_dim=256,
    intermediate_size=9216, num_decoder_layers=26,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
    model_quality=QualityMetricsCollection([MMLU(accuracy=51.3, shots=5), Hellaswag(accuracy=73.0, shots=10), MATH(accuracy=15.0, shots=4), GSM8K(accuracy=23.9, shots=5), TriviaQA(accuracy=59.4, shots=5), BIG_Bench(accuracy=41.9, shots=3)]),
)

# https://huggingface.co/google/gemma-2-9b/blob/main/config.json
gemma2_9b_config = ModelConfig(model='google/gemma-2-9B',
    hidden_size=3584, num_attention_heads=16, num_ffi = 2,
    num_key_value_heads=8, head_dim=256,
    intermediate_size=14336, num_decoder_layers=42,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
    model_quality=QualityMetricsCollection([MMLU(accuracy=71.3, shots=5), Hellaswag(accuracy=81.9, shots=10), MATH(accuracy=36.6, shots=4), GSM8K(accuracy=68.6, shots=5), TriviaQA(accuracy=76.6, shots=5), BIG_Bench(accuracy=68.2, shots=3)]),
)

# https://huggingface.co/google/gemma-2-27b-it/blob/main/config.json
gemma2_27b_config = ModelConfig(model='google/gemma-2-27B',
    hidden_size=4608, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=16, head_dim=128,
    intermediate_size=36864, num_decoder_layers=46,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
    model_quality=QualityMetricsCollection([MMLU(accuracy=75.2, shots=5), Hellaswag(accuracy=86.4, shots=10), MATH(accuracy=42.3, shots=4), GSM8K(accuracy=74.0, shots=5), TriviaQA(accuracy=83.7, shots=5), BIG_Bench(accuracy=74.9, shots=3)]),
)

# TODO: Interleaving Global and Local attention: https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf
# https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json
gemma3_1b_config = ModelConfig(model='google/gemma-3-1B',
    hidden_size=1152, num_attention_heads=4, num_ffi = 2,
    num_key_value_heads=1, head_dim=256,
    intermediate_size=6912, num_decoder_layers=26,
    vocab_size=262144, max_model_len=32*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=512,
model_quality=QualityMetricsCollection([                                  Hellaswag(accuracy=62.3, shots=10),                                                                  TriviaQA(accuracy=39.8, shots=5), BIG_Bench(accuracy=41.9, shots=3)]),
)

gemma3_4b_config = ModelConfig(model='google/gemma-3-4B',
    hidden_size=2560, num_attention_heads=16, num_ffi = 2,
    num_key_value_heads=4,
    intermediate_size=10240, num_decoder_layers=34,
    vocab_size=262144, max_model_len=128*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=1024,
    model_quality=QualityMetricsCollection([MMLU(accuracy=59.6, shots=5), Hellaswag(accuracy=77.2, shots=10), MATH(accuracy=24.2, shots=4), GSM8K(accuracy=38.4, shots=8), TriviaQA(accuracy=65.8, shots=5), BIG_Bench(accuracy=41.9, shots=3)]),
)

gemma3_12b_config = ModelConfig(model='google/gemma-3-12B',
    hidden_size=2304, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=4, head_dim=256,
    intermediate_size=9216, num_decoder_layers=26,
    vocab_size=262144, max_model_len=128*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
    model_quality=QualityMetricsCollection([MMLU(accuracy=74.5, shots=5), Hellaswag(accuracy=84.2, shots=10), MATH(accuracy=43.3, shots=4), GSM8K(accuracy=71.0, shots=8), TriviaQA(accuracy=78.2, shots=5), BIG_Bench(accuracy=41.9, shots=3)]),
)

# https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json
gemma3_27b_config = ModelConfig(model='google/gemma-3-27B',
    hidden_size=5376, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=16, head_dim=128,
    intermediate_size=21504, num_decoder_layers=62,
    vocab_size=262144, max_model_len=128*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=1024,
    model_quality=QualityMetricsCollection([MMLU(accuracy=78.6, shots=5), Hellaswag(accuracy=85.6, shots=10), MATH(accuracy=50, shots=4), GSM8K(accuracy=82.6, shots=8), TriviaQA(accuracy=85.5, shots=5), BIG_Bench(accuracy=41.9, shots=3)]),
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