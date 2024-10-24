import numpy as np
from GenZ.operator_base import Operator
import ast
from GenZ.Models import OpType, CollectiveType

class FC(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        input_a = (B, I)
        input_w = (O, I)
        output = (B, O)
        return input_a, input_w, output

    def get_gemms(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        left = B
        upper = O
        contract = I
        outer = 1
        return left, upper, contract, outer

    def get_num_ops(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, O, I])

class CONV1D(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 5

    def get_tensors(self):
        # B -> Batch Size
        # OF -> Number of output channels
        # IF -> Number of input channels
        # N -> Number of elements in the input
        # C -> Number of elements in the kernel
        B, OF, IF, N, C = self.dim[:self.get_effective_dim_len()]
        input_a = (B, N, IF)
        input_w = (OF, 1 , C)
        output = (B, N, OF)
        return input_a, input_w, output

    def get_gemms(self):
        B, OF, IF, N, C = self.dim[:self.get_effective_dim_len()]
        left = B*IF*N
        upper = OF
        contract = C*OF
        outer = 1
        return left, upper, contract, outer


    def get_num_ops(self):
        B, OF, IF, N, C = self.dim[:self.get_effective_dim_len()]
        return  np.prod([B, OF, N, C])

class CONV2D(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 7

    def get_tensors(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        input_a = (B, C, Y, X)
        input_w = (K, C, R, S)
        output = (B, K, Y, X)
        return input_a, input_w, output

    def get_gemms(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        left = B*Y*X
        upper = K
        contract = C*R*S
        outer = 1
        return left, upper, contract, outer


    def get_num_ops(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        return  np.prod([B, K, C, Y, X, R, S])

class GEMM(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 4

    def get_dimensions(self):
        return [self.get_tensors()]

    def get_tensors(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = (B, K, N)
        input_w = (M, K)
        output = (B, M, N)
        # print(input_a,input_w,output)
        return input_a, input_w, output


    def get_gemms(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        left = N
        upper = M
        contract = K
        outer = B
        return left, upper, contract, outer


    def get_num_ops(self):
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        return  np.prod([B, M, N, K])


class Logit(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 6

    def get_tensors(self):
        ## Refer FLAT paper for more detailed explanation on these parameters
        # B -> Batch Size
        # H -> Number of Heads
        # M -> Seq Len for Q
        # N -> Seq Len for K
        # D -> Split hidden size of key, query and values
        # Hkv -> Number of Heads for K and V.

        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        assert H % Hkv == 0 , f"H:{H} must be divisible by Hkv:{Hkv} "
        input_a = (B, H, M, D)
        input_w = (B, Hkv, N, D)
        output = (B, H, M, N)
        return input_a, input_w, output

    def get_gemms(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        left = M
        upper = N
        contract = D
        outer =B*H
        return left, upper, contract, outer


    def get_num_ops(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        return  np.prod([B, H, M, N, D])

class Attend(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 6

    def get_tensors(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        assert H % Hkv == 0 , "H must be divisible by Hkv"
        input_a = (B, H, M, N)
        input_w = (B, Hkv, N, D)
        output = (B, H, M, D)
        return input_a, input_w, output

    def get_gemms(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        left = M
        upper = D
        contract = N
        outer = B*H
        return left, upper, contract, outer

    def get_num_ops(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, H, M, N, D])

class DWCONV(Operator):
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 7

    def get_tensors(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        input_a = (B, C, Y, X)
        input_w = (C, R, S)
        output = (B, C, Y, X)
        return input_a, input_w, output

    def get_gemms(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        left = B*Y*X
        upper = 1
        contract = C*R*S
        outer = 1
        return left, upper, contract, outer

    def get_num_ops(self):
        B, K, C, Y, X, R, S = self.dim[:self.get_effective_dim_len()]
        return  np.prod([B, C, Y, X, R, S])

class Sync(Operator):   ## Just data movement.
    def __init__(self, dim, density):
        self.collective_type = dim[-2]
        self.num_collective_nodes = dim[-3]
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        return 0, 0, 0

    def get_dimensions(self):
        if self.num_collective_nodes == 1 and self.collective_type != CollectiveType.MessagePass:
            return None
        else:
            B, M, N = self.dim[:self.get_effective_dim_len()]
            return (B,M,N)

    def communication_data(self):
        if self.num_collective_nodes == 1 and self.collective_type != CollectiveType.MessagePass:
            return 0
        else:
            B, M, N = self.dim[:self.get_effective_dim_len()]
            return B*M*N

    def get_gemms(self):
        left = 0
        upper = 0
        contract = 0
        outer = 0
        return left, upper, contract, outer


    def get_num_ops(self):
        return 0

class Einsum(Operator):
    def __init__(self, dim, density):
        """
        equation: Einstein summation notation string
        dims: Dictionary of tensor dimensions keyed by the corresponding label in the equation
        """
        self.batch = dim[0]
        self.equation = dim[1]

        self.dimensions = {k: int(v) if isinstance(v, (int, float)) else v for k, v in ast.literal_eval(dim[2]).items()}
        for k, v in self.dimensions.items():
            if v == 'b':
                self.dimensions[k] = self.batch
            elif isinstance(v, str):
                raise ValueError(f"Unknown dimension {k}:{v} in equation {self.equation}")

        for var in set(''.join(self.equation.split('->')[0].split(','))):
            if var not in set(self.dimensions.keys()):
                raise ValueError(f"Invalid variable {var} in equation {self.equation}")

        super().__init__(dim=dim, density=density)


    def get_tensors(self):
        input_dims = self.equation.split('->')[0]
        input_a = [self.dimensions[label] for label in input_dims.split(',')[0]]
        input_b = [self.dimensions[label] for label in input_dims.split(',')[1]]
        output = [self.dimensions[label] for label in self.equation.split('->')[1]]
        return input_a, input_b, output

    def get_num_ops(self):
        """
        Compute the number of operations needed for the given einsum configuration.
        """
        input_dims = self.equation.split('->')[0]
        dim_labels = set(''.join(input_dims.split(',')))

        # The number of operations is the product of the dimensions involved in the contraction
        num_ops = np.prod([self.dimensions[label] for label in dim_labels])
        return num_ops

class Repeat(Operator):   ## Layer/Model Repetition
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        return 0, 0, 0
    def get_gemms(self):
        return 0, 0, 0, 0

    def get_dimensions(self):
        _, repeat_count, ID = self.dim[:self.get_effective_dim_len()]
        return repeat_count

    def get_num_ops(self):
        return 0

class EndRepeat(Operator):   ## Layer/Model Repetition
    def __init__(self, dim, density):
        super().__init__(dim=dim, density=density)

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        return 0, 0, 0

    def get_dimensions(self):
        _, repeat_count, ID = self.dim[:self.get_effective_dim_len()]
        return repeat_count

    def get_gemms(self):
        return 0, 0, 0, 0

    def get_num_ops(self):
        return 0