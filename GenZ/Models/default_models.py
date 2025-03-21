import numpy as np
from math import ceil, lcm
import sys
from typing import Optional
import inspect
from .model_quality import QualityMetricsCollection

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
        sliding_window=None,
        ffn_implementation="default",
        # MoE specific parameters
        moe_layer_freq = None,
        num_experts = 1,
        expert_top_k = 1,
        moe_intermediate_size = None,
        n_shared_experts = 0,
        shared_expert_intermediate_size = None,
        first_k_dense_replace = None,
        # Mamba specific parameters
        mamba_d_state=None,
        mamba_d_conv=None,
        mamba_expand=None,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        # Multi-Type model parameters
        mamba_layer_period=1,
        attn_layer_offset = 0,
        attn_layer_period = 1,
        expert_layer_offset = 0,
        expert_layer_period = 1,
        # Quality of Model
        model_quality: Optional[QualityMetricsCollection] = None,
        **kwargs,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.max_model_len = max_model_len      ## Maximum length of the model
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_ffi = num_ffi
        self.hidden_act = hidden_act


        # Attention Parameters
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        # MoE Parameters
        self.is_moe = num_experts > 1
        self.moe_layer_freq = moe_layer_freq    ## If n, than every nth value is moe layer.
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.n_shared_experts = n_shared_experts
        self.shared_expert_intermediate_size = shared_expert_intermediate_size if shared_expert_intermediate_size is not None else intermediate_size
        self.first_k_dense_replace = first_k_dense_replace

        # Mamba Parameters
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand if mamba_expand is not None else 1
        self.mamba_dt_rank = ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.is_mamba = (mamba_d_state is not None)


        # Multi-Type Model Parameters
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset
        self.mamba_layer_period = mamba_layer_period

        # Create the 2D list of parameters
        # print(self.is_mamba, self.is_moe)
        self.layer_type = []
        if self.is_mamba and self.is_moe:
            self.unique_layers = lcm(self.mamba_layer_period, self.expert_layer_period, self.attn_layer_period)
        elif self.is_mamba:
            self.unique_layers = self.mamba_layer_period
        elif self.is_moe:
            self.unique_layers = self.expert_layer_period
        else:
            self.unique_layers = 1
        num_repeats = self.num_decoder_layers / self.unique_layers
        assert num_repeats.is_integer(), "Number of decoder layers must be divisible by the unique layers"
        for i in range(self.unique_layers):
            # Determine the attention type
            if self.is_mamba and (i % self.mamba_layer_period == 0):
                attention_type = "Mamba"
            elif (i % self.attn_layer_period == self.attn_layer_offset):
                attention_type = "MHA-global"
            else:
                attention_type = "MHA-global"

            if self.is_moe and self.expert_layer_period and (i % self.expert_layer_period == self.expert_layer_offset):
                layer_type = "MoE"
            else:
                layer_type = "Dense"

            self.layer_type.append([attention_type, layer_type])

        self.ffn_implementation = ffn_implementation


        # Quality of Model
        self.model_quality = model_quality

        super().__init__()

    @property
    def layers_block_type(self):
        return [
            "MHA-global" if i % self.attn_layer_period == self.attn_layer_offset else "Mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]

    def get_kv_size(self):
        return 2*self.num_decoder_layers*self.head_dim*self.num_key_value_heads

    def __str__(self):
        return str(vars(self))

def get_all_model_configs(file_name):
    current_module = sys.modules[file_name]
    model_configs = {}
    for name, obj in inspect.getmembers(current_module):
        if isinstance(obj, ModelConfig):
            model_configs[obj.model] = obj
            if "/" in obj.model:
                model_configs[obj.model.split('/')[1]] = obj
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