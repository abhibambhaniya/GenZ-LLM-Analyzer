import pandas as pd
import os
from math import ceil
import numpy as np

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
        **kwargs,
    ):
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
        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))
      


def get_configs(name, return_full = False, get_model_config=False):
    name = name.lower()
    if  name in ['gpt-2', 'opt_125m']:      ## ViT,GPT-2 also has same
        model_config = ModelConfig(
        hidden_size=768, num_attention_heads=12, num_ffi = 1,
        intermediate_size=4*768, num_decoder_layers=12,
        )
    elif name == 'opt_350m' :
        model_config = ModelConfig(
        hidden_size=1024, num_attention_heads=16, num_ffi = 1,
        intermediate_size=4*1024, num_decoder_layers=24,
        )
    elif name == 'gpt-3_1b' or name == 'opt_1b' :
        model_config = ModelConfig(
        hidden_size=2048, num_attention_heads=32, num_ffi = 1,
        intermediate_size=4*2048, num_decoder_layers=24,
        )
    elif name == 'gpt-3_7b' or name == 'opt_7b' :
        model_config = ModelConfig(
        hidden_size=4096, num_attention_heads=32, num_ffi = 1,
        intermediate_size=4*4096, num_decoder_layers=32,
        )
    elif name == 'gpt-3_13b':
        model_config = ModelConfig(
        hidden_size=5140, num_attention_heads=40, num_ffi = 1,
        intermediate_size=4*5140, num_decoder_layers=40,
        )
    elif name == 'gpt-3' or name == 'opt_175b':
        model_config = ModelConfig(
        hidden_size=12288, num_attention_heads=96, num_ffi = 1,
        intermediate_size=4*12288, num_decoder_layers=96,
        )
    elif name == 'palm':
        model_config = ModelConfig(
            hidden_size=18432, num_attention_heads=48, num_ffi = 1,
            intermediate_size=4*18432, num_decoder_layers=118
            )
    elif name == 'gemma_7b':
        model_config = ModelConfig(
            hidden_size=3072, num_attention_heads=12, num_ffi = 2,
            intermediate_size=24576, num_decoder_layers=28, head_dim=256
            )
    elif name == 'llama_7b':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32, num_ffi = 2,
            intermediate_size=11008, num_decoder_layers=32
            )
    elif name == 'llama3_8b':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32,
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=14336, num_decoder_layers=32,
            )
    elif name == 'llama_13b':
        model_config = ModelConfig(
            hidden_size=5120, num_attention_heads=40, num_ffi = 2,
            intermediate_size=13824, num_decoder_layers=40
            )
    elif name == 'llama_33b':
        model_config = ModelConfig(
            hidden_size=6656, num_attention_heads=52, num_ffi = 2,
            intermediate_size=17888, num_decoder_layers=60
            )
    elif name == 'opt_30b':
        model_config = ModelConfig(
            hidden_size=7168, num_attention_heads=56, 
            intermediate_size=4*7168, num_decoder_layers=48,
            )
    elif name == 'llama_70b':
        model_config = ModelConfig(
            hidden_size=8192, num_attention_heads=64, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=28672, num_decoder_layers=80,
            )
    elif name == 'mixtral_7x8':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=14336, num_decoder_layers=32,
            expert_top_k=2, num_experts=8, moe_layer_freq=1
            )
    elif name == 'dbrx':
        model_config = ModelConfig(
            hidden_size=6144, num_attention_heads=48, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=10752, num_decoder_layers=40,
            expert_top_k=4, num_experts=16, moe_layer_freq=1
            )
    elif name == 'gpt-4':
        x = 84
        model_config = ModelConfig(
            hidden_size=84*128, num_attention_heads=84, 
            num_key_value_heads=84, num_ffi = 1,
            intermediate_size=4*84*128, num_decoder_layers=128,
            expert_top_k=2, num_experts=16, moe_layer_freq=1
            )
    elif name == 'grok-1':
        model_config = ModelConfig(
            hidden_size=6144, num_attention_heads=48, 
            num_key_value_heads=8, num_ffi = 1,
            intermediate_size=8*6144, num_decoder_layers=64,
            expert_top_k=2, num_experts=8, moe_layer_freq=1
            )
    elif name == 'super_llm':
        x = 108
        model_config = ModelConfig(
            hidden_size=x*128, num_attention_heads=x, 
            num_key_value_heads=x, num_ffi = 2,
            intermediate_size=4*x*128, num_decoder_layers=128,
            expert_top_k=4, num_experts=32, moe_layer_freq=1
            )
    else:
        ## If unknown name, then giving parameters of BERT
        print("ERROR, model name parsed incorrect, please check!!! Model Name:",name)
    
    return model_config

def create_inference_moe_prefix_model(input_sequence_length, name='BERT', data_path='./', masked=False,
                         output_gen_tokens=32, **args):
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    model_config = get_configs(name, get_model_config=True)
    
    M = N  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    MQA = ( Hkv != H)

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}' 
    H = max(ceil(H/tensor_parallel),1)
    Hkv = max(ceil(Hkv/tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []

   
    query =         [[D//tensor_parallel + 2*Hkv*Dq, N, D, 1, 1, 1, 3]]
 
    logit =         [[H, M, N, Dq, Hkv, 1, 7 if MQA else 4]]
    attend =        [[H, M, N, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, M, D//tensor_parallel, 1, 1, 1, 3]]

    if moe_layer_freq:
        num_tokens_per_expert = M*K // E
        ffup =           [[E*Df, num_tokens_per_expert, D, 1, 1, 1, 3]]
        ffdown =           [[D, num_tokens_per_expert, E*Df, 1, 1, 1, 3]]
    else:
        ffup =           [[Df, M, D, 1, 1, 1, 3]]
        ffdown =           [[D, M, Df, 1, 1, 1, 3]]



    layers = query + logit + attend + output 
    

    for _ in range(fi):
        layers += ffup
    layers += ffdown

     
    # densities = np.ones((len(layers), 3), dtype=float) 
    
    # df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    # df.to_csv(os.path.join(sparsity_file_path, name+ '_decode' + '.csv'),  header=True, index=None)

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_prefix'+'.csv'),  header=True, index=None)

    return name+'_prefix'

def create_inference_moe_decode_model(input_sequence_length, name='BERT', data_path='./',
                         output_gen_tokens=32, **args):
    
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    
    model_config = get_configs(name, get_model_config=True)
    
    N  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    MQA = ( Hkv != H)
    
    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}' 

    H = max(ceil(H/tensor_parallel),1)
    Hkv = max(ceil(Hkv/tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []


    query =         [[D//tensor_parallel + 2*Hkv*Dq, 1, D, 1, 1, 1, 3]]

    logit_pre =         [[H, 1, N, Dq, Hkv, 1, 7 if MQA else 9]]
    attend_pre =        [[H, 1, N, Dq, Hkv, 1, 8 if MQA else 10]]
    logit_suf =         [[H, 1, output_gen_tokens, Dq, Hkv, 1, 7 if MQA else 4]]
    attend_suf =        [[H, 1, output_gen_tokens, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, 1, D//tensor_parallel, 1, 1, 1, 3]]
    ffup =           [[K*Df, 1, D, 1, 1, 1, 3]]    ## Df is already divided
    ffdown =           [[D, 1, K*Df, 1, 1, 1, 3]]

    ffup_unused =   [[(E-K)*Df, 0, D, 1, 1, 1, 3]]   
    ffdown_unused =   [[D, 0, (E-K)*Df, 1, 1, 1, 3]] 

    layers = query + logit_pre + logit_suf + attend_pre + attend_suf + output
    
    
    for _ in range(fi):
        layers += (ffup + ffup_unused) if moe_layer_freq  else ffup
    layers += (ffdown + ffdown_unused)  if moe_layer_freq  else ffdown
        
    # densities = np.ones((len(layers), 3), dtype=float) 
    
    # df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    # df.to_csv(os.path.join(sparsity_file_path, name+ '_decode' + '.csv'),  header=True, index=None)

    

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_decode'+'.csv'),  header=True, index=None)
    
    return name+'_decode'
