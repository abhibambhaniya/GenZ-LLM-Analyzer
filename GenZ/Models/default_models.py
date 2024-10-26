import numpy as np
from math import ceil
import sys
import inspect

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
        max_model_len = 128000,
        hidden_size=4096,
        intermediate_size=11008,
        num_ffi = 1,    ## Number of feed forward parallel in the first up projection
        num_encoder_layers=0,
        num_decoder_layers=32,
        num_attention_heads=32,
        head_dim=None,
        num_key_value_heads=None,
        hidden_act="silu",
        sliding_window=4096,
        # MoE specific parameters
        moe_layer_freq = None,
        num_experts = 1,
        expert_top_k = 1,
        # Mamba specific parameters
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
        self.sliding_window = sliding_window
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

        self.max_model_len = max_model_len      ## Maximum length of the model

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        super().__init__()

    def __str__(self):
        return str(vars(self))

def get_all_model_configs():
    current_module = sys.modules[__name__]
    model_configs = {}
    for name, obj in inspect.getmembers(current_module):
        if isinstance(obj, ModelConfig):
            model_configs[obj.model] = obj
    return model_configs

class ModelCollection():
    def __init__(self, models=None):
        if models is not None:
            self.models = models
        else:
            self.models = {}

    def add_model(self, model_config):
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be an instance of ModelConfig")
        self.models[model_config.model] = model_config

    def get_model(self, model_name):
        model_name_lower = model_name.lower()
        for name in self.models:
            if name.lower() == model_name_lower:
                return self.models[name]
        return None

    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]

    def list_models(self):
        unique_models = {}
        for key, value in self.models.items():
            if value not in unique_models.values():
                unique_models[key] = value
            else:
                existing_key = next(k for k, v in unique_models.items() if v == value)
                if len(key) > len(existing_key):
                    del unique_models[existing_key]
                    unique_models[key] = value
        return list(unique_models.keys())

    def add_model_collection(self, model_collection):
        if not isinstance(model_collection, ModelCollection):
            raise TypeError("model_collection must be an instance of ModelCollection")
        self.models.update(model_collection.models)

    def __str__(self):
        return str({name: str(config) for name, config in self.models.items()})

from .Model_sets.alibaba import alibaba_models
from .Model_sets.google import google_models
from .Model_sets.microsoft import microsoft_models
from .Model_sets.misc import misc_models
from .Model_sets.mistral import mistral_models
from .Model_sets.meta import meta_models
from .Model_sets.nvidia import nvidia_models

MODEL_DICT = ModelCollection()
MODEL_DICT.add_model_collection(ModelCollection(alibaba_models))
MODEL_DICT.add_model_collection(ModelCollection(google_models))
MODEL_DICT.add_model_collection(ModelCollection(microsoft_models))
MODEL_DICT.add_model_collection(ModelCollection(misc_models))
MODEL_DICT.add_model_collection(ModelCollection(mistral_models))
MODEL_DICT.add_model_collection(ModelCollection(meta_models))
MODEL_DICT.add_model_collection(ModelCollection(nvidia_models))