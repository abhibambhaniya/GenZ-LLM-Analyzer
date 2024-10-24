from enum import IntEnum

class OpType(IntEnum):
    FC = 0
    CONV2D = 1
    DWCONV = 2
    GEMM = 3
    Logit = 4
    Attend = 5
    Sync = 6
    Logit_BM_PREFILL = 9
    Attend_BM_PREFILL = 10
    CONV1D = 11
    EINSUM = 12
    REPEAT = 13
    ENDREPEAT = 14

class ResidencyInfo(IntEnum):
    All_offchip = 0
    A_onchip = 1
    B_onchip = 2
    C_onchip = 3
    AB_onchip = 4
    AC_onchip = 5
    BC_onchip = 6
    All_onchip = 7

from enum import IntEnum

class CollectiveType(IntEnum):
    AllReduce = 1
    All2All = 2
    AllGather = 3
    ReduceScatter = 4
    MessagePass = 5

def parse_einsum_expression(expression, *tensors):
    einsum_vars = {}
    input_subscripts, output_subscript = expression.split('->')
    input_subscripts = input_subscripts.split(',')

    for tensor, subscripts in zip(tensors, input_subscripts):
        for dim, subscript in zip(tensor, subscripts):
            if subscript not in einsum_vars:
                einsum_vars[subscript] = dim
            elif einsum_vars[subscript] != dim:
                raise ValueError(f"Dimension mismatch for subscript: {subscript}, Got: {dim}, Expected: {einsum_vars[subscript]}")

    for subscript in output_subscript:
        if subscript not in einsum_vars:
            einsum_vars[subscript] = None

    return einsum_vars