import numpy as np
import math
from GenZ.unit import Unit
import json
class System(object):
    compute_multiplier = {'int8': 0.5, 'bf16': 1, 'f32': 2, 'int4': 0.25, 'int2':0.125, 'fp8': 0.5,  'fp6':0.5, 'fp4': 0.25}
    mem_multiplier = {'int8': 1, 'bf16': 2, 'f32': 4, 'int4':0.5, 'int2':0.25, 'fp8':1,  'fp6':0.75, 'fp4':0.5}
    def __init__(self, unit=None,
                flops=123, mxu_shape=None,
                onchip_mem_bw=18000, on_chip_mem_size=float('Inf'),
                offchip_mem_bw=900, off_chip_mem_size=float('Inf'),
                external_mem_bw=0,
                frequency=940, bits='bf16',
                compute_efficiency=1, memory_efficiency=1, comm_efficiency=1,
                interchip_link_bw = 25, num_nodes = 1, interchip_link_latency=1.9,
                compute_engine='GenZ',    # GenZ or Scale-sim
                collective_strategy='GenZ',    # GenZ or ASTRA-SIM
                topology='FullyConnected',
                parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
                network_config = None,
                gear_params = None,
                ):

        if unit is None:
            self.unit = Unit()
        else:
            self.unit = unit

        self.flops = self.unit.unit_to_raw(flops, type='C')
        self.op_per_sec = self.flops/2

        self.frequency = self.unit.unit_to_raw(frequency, type='F')
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')
        self.interchip_link_bw = self.unit.unit_to_raw(interchip_link_bw, type='BW')
        self.interchip_link_latency = interchip_link_latency * 1e-6     ## us
        self.external_mem_bw = self.unit.unit_to_raw(external_mem_bw, type='BW')
        self.on_chip_mem_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.on_chip_mem_left_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.off_chip_mem_size = self.unit.unit_to_raw(off_chip_mem_size, type='M')
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.comm_efficiency = comm_efficiency
        self.mxu_shape = mxu_shape

        self.compute_engine = compute_engine
        assert self.compute_engine in ['GenZ', 'Scale-sim'], "Invalid compute_engine. Must be one of: GenZ, Scale-sim"

        self.collective_strategy = collective_strategy
        assert self.collective_strategy in ['GenZ', 'ASTRA-SIM'], "Invalid collective_strategy. Must be one of: GenZ, ASTRA-SIM"
        self.num_nodes = num_nodes
        self.topology = topology
        self.bits = bits
        self.parallelism_heirarchy = parallelism_heirarchy   ## TP{1}_EP{1}_PP{1}
        self.network_config = network_config
        if gear_params:
            self.gear_r = gear_params['r']
            self.gear_s = gear_params['s']
            self.gear_b = gear_params['b']
            self.quantization_type = 'gear'
        else:
            self.quantization_type = None

    def __str__(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes} \n"
        b = f"On-Chip mem size: {unit.raw_to_unit(self.on_chip_mem_size, type='M')} MB , Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')} MB\n"
        c = f"On-Chip mem BW: {unit.raw_to_unit(self.onchip_mem_bw, type='BW')} GB/s , Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s,\n"
        return a+b+c

    def get_params(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes}"
        b = f" Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')/1024} GB "
        c = f" Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s"
        return a+b+c

    @classmethod
    def from_dict(cls, config_dict):
        init_params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        filtered_params = {k: v for k, v in config_dict.items() if k in init_params}
        return cls(**filtered_params)

    @classmethod
    def from_json(cls, json_str):
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def set_onchip_mem_bw(self,onchip_mem_bw):
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')

    def set_offchip_mem_bw(self,offchip_mem_bw):
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')

    def get_offchip_mem_bw(self):
        return self.unit.raw_to_unit(self.offchip_mem_bw,type='BW')

    def get_external_mem_bw(self):
        return self.unit.raw_to_unit(self.external_mem_bw,type='BW')

    def get_interchip_link_bw(self):
        return self.unit.raw_to_unit(self.interchip_link_bw,type='BW')

    def get_off_chip_mem_size(self):
        return self.unit.raw_to_unit(self.off_chip_mem_size,type='M')


    def claim_onchip_mem(self, data_sz):
        if data_sz > self.on_chip_mem_left_size:
            raise ValueError(f'Not enough on-chip memory: Need {data_sz}, only has {self.on_chip_mem_size}')
        self.on_chip_mem_left_size -= data_sz
        return self.on_chip_mem_left_size

    def release_onchip_mem(self, data_sz):
        self.on_chip_mem_left_size = max(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)
        return self.on_chip_mem_left_size

    def get_bit_multiplier(self, type='C', data='w', operators=None):
        if type == 'C':
            return self.compute_multiplier[self.bits]
        elif type == 'M':
            if self.quantization_type == 'gear':
                if data == 'k' or data == 'v':
                    # print(
                    #     f"Quantized KV bits: {self.mem_multiplier[self.gear_b]}",
                    #     f"Sparsity bits:{(self.gear_s/100) * self.mem_multiplier[self.bits]}",
                    #     f"Operators: ", operators,
                    #     f"Low Rank Bits: {((np.prod(operators[:-2])/np.prod(operators)) * (operators[-2]*self.gear_r + operators[-1]*self.gear_r) * self.mem_multiplier[self.bits])}")

                    return (    self.mem_multiplier[self.gear_b]
                                + (self.gear_s/100) * self.mem_multiplier[self.bits]
                                + ((np.prod(operators[:-2])/np.prod(operators)) * (operators[-2]*self.gear_r + operators[-1]*self.gear_r) * self.mem_multiplier[self.bits])
                    )
            return self.mem_multiplier[self.bits]