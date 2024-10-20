import numpy as np
from math import ceil


class ModelConfig():
    r"""
    This is the configuration class to store the configuration of a [`Model`]. It is used to instantiate an LLM
    model according to the specified arguments, defining the model architecture.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
    """
    def __init__(
        self,
        model = 'dummy',
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_ffi = 1,    ## Number of feed forward parallel in the first up projection
        num_encoder_layers=0,
        num_decoder_layers=32,
        num_attention_heads=32,
        head_dim=None,
        num_key_value_heads=None,
        moe_layer_freq = None,
        hidden_act="silu",
        num_experts = 1,
        expert_top_k = 1,
        max_model_len = 128000,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_ffi = num_ffi
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if head_dim is None:
            head_dim = self.hidden_size // self.num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.moe_layer_freq = moe_layer_freq    ## If n, than every nth value is moe layer.
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k

        self.max_model_len = max_model_len             ## TODO:Put real values

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))
    




# https://huggingface.co/facebook/opt-125m/blob/main/config.json
opt_125m_config = ModelConfig(model='facebook/opt-125M',
    hidden_size=768, num_attention_heads=12, num_ffi = 1,
    intermediate_size=4*768, num_decoder_layers=12,
)
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json
phi3mini_config = ModelConfig(model='microsoft/Phi-3-mini',
    hidden_size=3072, num_attention_heads=32, num_ffi = 2,
    intermediate_size=8192, num_decoder_layers=32,
)
# https://huggingface.co/microsoft/Phi-3-small-128k-instruct/blob/main/config.json
phi3small_config = ModelConfig(model='microsoft/Phi-3-small',
    hidden_size=4096, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=8, head_dim=128,
    intermediate_size=14336, num_decoder_layers=32,
)

# https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json
phi3medium_config = ModelConfig(model='microsoft/Phi-3-medium',
    hidden_size=5120, num_attention_heads=40, num_ffi = 2,
    num_key_value_heads=10, head_dim=128,
    intermediate_size=17920, num_decoder_layers=40,
)

# https://huggingface.co/openai-community/gpt2/blob/main/config.json
gpt2_config = ModelConfig(model='openai/gpt2',
    hidden_size=768, num_attention_heads=12, num_ffi = 1,
    intermediate_size=4*768, num_decoder_layers=12,
)

opt_350m_config = ModelConfig(model='facebook/OPT-350M',
    hidden_size=1024, num_attention_heads=16, num_ffi = 1,
    intermediate_size=4*1024, num_decoder_layers=24,
)

opt_1b_config = ModelConfig(model='facebook/OPT-1B',
    hidden_size=2048, num_attention_heads=32, num_ffi = 1,
    intermediate_size=4*2048, num_decoder_layers=24,
)

opt_7b_config = ModelConfig(model='facebook/OPT-7B',
    hidden_size=4096, num_attention_heads=32, num_ffi = 1,
    intermediate_size=4*4096, num_decoder_layers=32,
)

opt_13b_config = ModelConfig(model='facebook/OPT-13B',
    hidden_size=5140, num_attention_heads=40, num_ffi = 1,
    intermediate_size=4*5140, num_decoder_layers=40,
)

gpt3_config = ModelConfig(model='openai/GPT3-175B',
    hidden_size=12288, num_attention_heads=96, num_ffi = 1,
    intermediate_size=4*12288, num_decoder_layers=96,
)

palm_config = ModelConfig(model='google/palm',
        hidden_size=18432, num_attention_heads=48, num_ffi = 1,
        intermediate_size=4*18432, num_decoder_layers=118
)

# https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/config.json
falcon7b_config = ModelConfig(model='tiiuae/falcon-7b-instruct',
    hidden_size=4544, num_attention_heads=71, num_ffi = 1,
    num_key_value_heads=71, head_dim=64,
    intermediate_size=4544*4, num_decoder_layers=32,
)

# https://huggingface.co/google/gemma-2b-it/blob/main/config.json
gemma_2b_config = ModelConfig(model='google/gemma-2B',
    hidden_size=2048, num_attention_heads=8, num_ffi = 2,
    intermediate_size=16384, num_decoder_layers=18, head_dim=256
)
# https://huggingface.co/google/gemma-7b-it/blob/main/config.json
gemma_7b_config = ModelConfig(model='google/gemma-7B',
    hidden_size=3072, num_attention_heads=16, num_ffi = 2,
    intermediate_size=24576, num_decoder_layers=28, head_dim=256
)

# https://huggingface.co/google/gemma-2-9b/blob/main/config.json
gemma2_9b_config = ModelConfig(model='google/gemma-2-9B',
    hidden_size=3584, num_attention_heads=16, num_ffi = 2,
    num_key_value_heads=8, head_dim=256,
    intermediate_size=14336, num_decoder_layers=42,
)

# https://huggingface.co/google/gemma-2-27b-it/blob/main/config.json
gemma2_27b_config = ModelConfig(model='google/gemma-2-27B',
    hidden_size=4608, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=16, head_dim=128,
    intermediate_size=36864, num_decoder_layers=46,
)

# https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
llama_7b_config = ModelConfig(model='meta-llama/Llama-2-7B',
    hidden_size=4096, num_attention_heads=32, num_ffi = 2,
    intermediate_size=11008, num_decoder_layers=32
)

# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
llama3_8b_config = ModelConfig(model='meta-llama/Llama-3.1-8B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
)
# https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
llama_13b_config = ModelConfig(model='meta-llama/Llama-2-13B',
    hidden_size=5120, num_attention_heads=40, num_ffi = 2,
    intermediate_size=13824, num_decoder_layers=40
)

# https://huggingface.co/Secbone/llama-33B-instructed/blob/main/config.json
llama_33b_config = ModelConfig(model='meta-llama/Llama-33B',
    hidden_size=6656, num_attention_heads=52, num_ffi = 2,
    intermediate_size=17920, num_decoder_layers=60
)

opt_30b_config = ModelConfig(model='facebook/opt-30B',
    hidden_size=7168, num_attention_heads=56,
    intermediate_size=4*7168, num_decoder_layers=48,
)


# https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
llama_70b_config = ModelConfig(model='meta-llama/Llama-2-70B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=28672, num_decoder_layers=80,
)

# https://huggingface.co/meta-llama/Meta-Llama-3.1-405B
llama_405b_config = ModelConfig(model='meta-llama/Llama-3.1-405B',
    hidden_size=16384, num_attention_heads=128,
    num_key_value_heads=16, num_ffi = 2,
    intermediate_size=3.25*16384, num_decoder_layers=126,
)

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
mistral_7b_config = ModelConfig(model='mistralai/Mistral-7B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
)
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json
mixtral_8x7b_config = ModelConfig(model='mistralai/Mixtral-8x7B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
    expert_top_k=2, num_experts=8, moe_layer_freq=1
)

# https://huggingface.co/databricks/dbrx-base/blob/main/config.json
dbrx_config = ModelConfig(model='databricks/dbrx-base',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=10752, num_decoder_layers=40,
    expert_top_k=4, num_experts=16, moe_layer_freq=1
)

gpt_4_config = ModelConfig(model='openai/GPT-4',
    hidden_size=84*128, num_attention_heads=84,
    num_key_value_heads=84, num_ffi = 1,
    intermediate_size=4*84*128, num_decoder_layers=128,
    expert_top_k=2, num_experts=16, moe_layer_freq=1
)

# https://huggingface.co/xai-org/grok-1/blob/main/RELEASE
grok_1_config = ModelConfig(model='xai-org/grok-1',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 1,
    intermediate_size=8*6144, num_decoder_layers=64,
    expert_top_k=2, num_experts=8, moe_layer_freq=1
)

# https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/config.json
glm_9b_config = ModelConfig(model='THUDM/glm-4-9b-chat',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=2, num_ffi = 2,
    intermediate_size=13696, num_decoder_layers=40,
    max_model_len=131072, vocab_size=151552,
)

# https://huggingface.co/state-spaces/mamba-130m-hf/blob/main/config.json
mamba_130m_config = ModelConfig(model='state-spaces/mamba-130m-hf',
    hidden_size=768, num_attention_heads=1,
    num_key_value_heads=1, num_ffi = 2,
    intermediate_size=1536, num_decoder_layers=24,
    max_model_len=131072, vocab_size=50280,
    mamba_d_state = 16, mamba_dt_rank = 48, mamba_expand = 2, mamba_d_conv=4,
)

super_llm_config = ModelConfig(model='SuperLLM-10T',
    hidden_size=108*128, num_attention_heads=108,
    num_key_value_heads=108, num_ffi = 2,
    intermediate_size=4*108*128, num_decoder_layers=128,
    expert_top_k=4, num_experts=32, moe_layer_freq=1
)

MODEL_DICT = {
    'opt_125m': opt_125m_config,
    'facebook/opt-125m': opt_125m_config,
    'opt_350m': opt_350m_config,
    'facebook/opt-350m': opt_350m_config,
    'opt_1b': opt_1b_config,
    'facebook/opt-1.3b': opt_1b_config,
    'opt_7b': opt_7b_config,
    'facebook/opt-7b': opt_7b_config,
    'opt_13b': opt_13b_config,
    'facebook/opt-13b': opt_13b_config,
    'opt_175b': gpt3_config,
    'facebook/opt-175b': gpt3_config,
    'opt_30b': opt_30b_config,
    'facebook/opt-30B': opt_30b_config,
    'phi3mini': phi3mini_config,
    'microsoft/phi3mini': phi3mini_config,
    'phi3small': phi3small_config,
    'microsoft/phi3small': phi3small_config,
    'phi3medium': phi3medium_config,
    'microsoft/phi3medium': phi3medium_config,
    'gpt-2': gpt2_config,
    'openai-community/gpt2': gpt2_config,
    'gpt-3_1b': opt_1b_config,
    'gpt-3_7b': opt_7b_config,
    'gpt-3': gpt3_config,
    'openai/gpt-3': gpt3_config,
    'palm': palm_config,
    'google/palm': palm_config,
    'falcon7b': falcon7b_config,
    'tiiuae/falcon-7b-instruct': falcon7b_config,
    'gemma_2b': gemma_2b_config,
    'google/gemma-2b-it': gemma_2b_config,
    'gemma_7b': gemma_7b_config,
    'google/gemma-7b-it': gemma_7b_config,
    'gemma2_9b': gemma2_9b_config,
    'google/gemma-2-9b': gemma2_9b_config,
    'gemma2_27b': gemma2_27b_config,
    'google/gemma-2-27b-it': gemma2_27b_config,
    'llama_7b': llama_7b_config,
    'meta-llama/llama-2-7b': llama_7b_config,
    'llama3_8b': llama3_8b_config,
    'meta-llama/meta-llama-3.1-8b': llama3_8b_config,
    'llama_13b': llama_13b_config,
    'meta-llama/llama-2-13b': llama_13b_config,
    'llama_33b': llama_33b_config,
    'meta-llama/llama-2-33b': llama_33b_config,
    'llama_70b': llama_70b_config,
    'meta-llama/llama-2-70b': llama_70b_config,
    'llama_405b': llama_405b_config,
    'meta-llama/meta-llama-3.1-405b': llama_405b_config,
    'mistral_7b': mistral_7b_config,
    'mistralai/mistral-7b': mistral_7b_config,
    'mixtral_8x7b': mixtral_8x7b_config,
    'mistralai/mixtral-8x7b': mixtral_8x7b_config,
    'dbrx': dbrx_config,
    'databricks/dbrx-base': dbrx_config,
    'gpt-4': gpt_4_config,
    'openai/gpt-4': gpt_4_config,
    'grok-1': grok_1_config,
    'xai-org/grok-1': grok_1_config,
    'glm-9b': glm_9b_config,
    'THUDM/glm-4-9b-chat': glm_9b_config,
    'state-spaces/mamba-130m-hf': mamba_130m_config,
    'super_llm': super_llm_config,
}