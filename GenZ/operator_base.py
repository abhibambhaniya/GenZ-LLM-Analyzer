import numpy as np
from operator import mul
from math import ceil
from GenZ.unit import Unit

from GenZ.Models import OpType, CollectiveType
from GenZ.collective_times import get_AR_time, get_A2A_time, get_message_pass_time

# 4, 5 Regular Logit and Attend
# 9, 10 Beam Merge Logit and attend
op_type_dicts = {0: 'FC', 1: 'CONV2D', 2: 'DWCONV', 3: 'GEMM', 4: 'Logit', 5: 'Attend', 6:'Sync',
                9:'Logit', 10:'Attend', 11:'CONV1D', 12:'Einsum', 13:'Repeat', 14:'EndRepeat'}
class Operator(object):
    def __init__(self, dim, density=(1.0,1.0,1.0)):
        self.dim = [int(x) if isinstance(x, (int, float, np.int32, np.int64)) else x for x in dim]
        self.density_a, self.density_w, self.density_o = density
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()
        self.set_mem_pin(*self.get_default_mem_loc())

    def get_default_mem_loc(self):
        return ['off', 'off', 'off']

    def set_mem_pin(self, input_a=None, input_b=None, output=None):
        if input_a is not None:
            self.input_a_loc = input_a
        if input_b is not None:
            self.input_w_loc = input_b
        if output is not None:
            self.output_loc = output

    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output

    def get_density_list(self):
        return [self.density_a, self.density_w, self.density_o]

    def get_op_type(self, dim):
        return op_type_dicts[dim[-1]]

    def get_tensors(self):
        pass

    def get_size(self, tensor):
        return np.prod(tensor)

    # Each kind of operation function will have its own num ops, in which using the layer parameters obtained from the
    # .csv file it will give out number of required ops .
    def get_num_ops(self):
        pass

    def get_dimensions(self):
        return self.get_tensors()

    # For each kind of operator, this returns number of required paramters for that layer type. (Refer operators.py )
    def get_effective_dim_len(self):
        pass

    def get_num_data(self):
        return sum(self.get_sz_list())

    def get_effective_num_data(self, system):
        return sum(self.get_sz_list())


    def get_ideal_memory_time(self, system):
        sz_list = self.get_sz_list()
        memory_time_onchip = 0
        memory_time_offchip = 0
        for tensor_sz in sz_list:
            memory_time_onchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.onchip_mem_bw
            memory_time_offchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.offchip_mem_bw
        return  memory_time_offchip, memory_time_onchip


    def get_compute_time(self, system):
        return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec


    def get_effective_num_ops(self, system=None):
        return self.get_num_ops()


# The function returns the size of each of the 3 models parameter for each layer, i.e. input, weights and outputs.
    def get_sz_list(self):
        return list(map(self.get_size, [self.input_a, self.input_w, self.output]))

    def get_loc_list(self):
        return [self.input_a_loc, self.input_w_loc, self.output_loc]

    def get_memory_time(self, system):
        sz_list = self.get_sz_list()
        loc_list = self.get_loc_list()
        memory_time = 0
        ## Assume infinite memory
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'off':
                bw = system.offchip_mem_bw
            elif loc == 'on':
                bw = system.onchip_mem_bw
            else:
                raise ValueError(f'Wrong bw allocation: {loc}.')
            memory_time += tensor_sz * system.get_bit_multiplier(type='M', data='w' if self.get_op_type(self.dim) == 'GEMM' else 'a')/bw
        return memory_time

    def get_communication_time(self, system):
        '''
            Returns the communication time for the operator in seconds.
        '''
        if self.get_op_type(self.dim) != 'Sync':
            return 0
        else:
            data_size = self.communication_data() * system.get_bit_multiplier(type='M', data='a')
            if self.collective_type == CollectiveType.AllReduce:
                return get_AR_time(data_size , self.num_collective_nodes, system) / 1000
            elif  self.collective_type == CollectiveType.All2All:
                return get_A2A_time(data_size , self.num_collective_nodes, system) / 1000
            elif  self.collective_type == CollectiveType.MessagePass:
                return get_message_pass_time(data_size, system) / 1000
            else:
                raise ValueError(f'Unknown collective type: {self.collective_type}.')

    def get_onchip_occupancy(self):
        sz_list = self.get_sz_list()
        loc_list = self.get_loc_list()
        onchip_mem_occupancy = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'on':
                onchip_mem_occupancy += tensor_sz

        return onchip_mem_occupancy

    def get_model_characterstics(self, system, unit = Unit()):
        num_ops =  self.get_num_ops()
        num_data = self.get_num_data() * system.get_bit_multiplier(type='M')
        op_intensity = num_ops/num_data  if num_data else 0
        input_a_size, input_w_size, output_size = self.get_sz_list()
        ret = {
            'Op Type': self.get_op_type(self.dim),
            'Dimension': self.get_dimensions(),
            'Op Intensity': op_intensity,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(sum(self.get_sz_list()), type='M')* system.get_bit_multiplier(type='M'),
        }

        return ret

    def get_roofline(self, system, unit):
        ideal_complete_offchip_time, ideal_complete_onchip_time = self.get_ideal_memory_time(system=system)
        # x2 for ops -> MAC has 1 multiplication and 1 Addition hence 2.
        num_ops = self.get_effective_num_ops(system) * 2
        num_data = self.get_effective_num_data(system) * system.get_bit_multiplier(type='M')
        op_intensity = num_ops/num_data if num_data else 0

        compute_time = self.get_compute_time(system=system)

        compute_time /= system.compute_efficiency
        compute_efficiency = system.compute_efficiency

        memory_time = self.get_memory_time(system=system) / system.memory_efficiency

        comm_time = self.get_communication_time(system=system) / system.comm_efficiency
        if compute_time == 0:
            memory_time = 0
        exec_time = max(compute_time, memory_time, comm_time)
        thrpt = num_ops/exec_time if exec_time else 0
        com_to_mem_ratio = compute_time/memory_time if memory_time else 0
        if com_to_mem_ratio == 0:
            boundedness = 'Collective'
        else:
            boundedness = 'Compute' if com_to_mem_ratio > 1 else 'Memory'

        input_a_size, input_w_size, output_size = self.get_sz_list()

        ret = {
            'Op Type': self.get_op_type(self.dim),
            'Dimension': self.get_dimensions(),
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'C Effcy': compute_efficiency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M')* system.get_bit_multiplier(type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(sum(self.get_sz_list()), type='M')* system.get_bit_multiplier(type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Compute time ({unit.unit_time})': unit.raw_to_unit(compute_time, type='T'),
            f'Memory time ({unit.unit_time})': unit.raw_to_unit(memory_time, type='T'),
            f'Communication time ({unit.unit_time})': unit.raw_to_unit(comm_time, type='T'),
            f'Compute cycle': compute_time*system.frequency,
            f'Memory cycle': memory_time*system.frequency,
            f'Communication cycle': comm_time*system.frequency,
        }

        return ret










