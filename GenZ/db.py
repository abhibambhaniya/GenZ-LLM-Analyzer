"""RuntimeDB — thin wrapper around aiconfigurator's PerfDatabase.

Instead of maintaining a separate SQLite database, we delegate all operator
runtime lookups to the aiconfigurator SDK which already provides:
  - Real silicon measurements for H100, A100, L40S, B200, etc.
  - Bi-linear and 3D interpolation across (batch_size, seq_len, etc.)
  - Fallback modes (SILICON → HYBRID → EMPIRICAL → SOL)
  - NCCL collective measurements
  - MoE, attention, GEMM, and other operator types

Latencies returned by aiconfigurator are in **milliseconds**.
This wrapper converts to **microseconds** for the simulator.
"""

import sys
import os
from GenZ.operator_base import Operator
from GenZ.Models import OpType, CollectiveType
from GenZ.system import System

# Make the aiconfigurator submodule importable without pip install.
_SUBMODULE_SRC = os.path.join(os.path.dirname(__file__), "aiconfigurator", "src")
if _SUBMODULE_SRC not in sys.path:
    sys.path.insert(0, _SUBMODULE_SRC)

from dataclasses import dataclass
from typing import Optional

from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    get_database,
    get_supported_databases,
    get_latest_database_version,
)
from aiconfigurator.sdk.common import (
    DatabaseMode,
    GEMMQuantMode,
    FMHAQuantMode,
    CommQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)

bits_to_gemm_quants = {
    'int8': GEMMQuantMode.int8_wo,
    'bf16': GEMMQuantMode.float16,
    'int4': GEMMQuantMode.int4_wo,
    'fp8': GEMMQuantMode.fp8,
    'fp4': GEMMQuantMode.nvfp4}

bits_to_comm_quants = {
    'int8': CommQuantMode.int8,
    'bf16': CommQuantMode.half,
    'fp8': CommQuantMode.fp8,
}  

@dataclass(frozen=True)
class OperatorTiming:
    """Result of an operator runtime lookup."""

    latency_us: float
    energy_wms: Optional[float] = None  # watt-milliseconds
    source: str = "silicon"


class RuntimeDB:
    """Operator runtime lookup backed by aiconfigurator PerfDatabase.

    Args:
        hardware: Hardware key (e.g. 'h100_sxm', 'a100_sxm', 'l40s').
        backend: Serving backend for measurements (default: 'vllm').
        version: Backend version. If None, uses latest available.
        database_mode: Query mode. SILICON uses real measurements.
    """

    def __init__(
        self,
        hardware: str,
        backend: str = "vllm",
        version: Optional[str] = None,
        database_mode: DatabaseMode = DatabaseMode.HYBRID,
        gemm_bits: str = 'fp8',
        comm_bits: str = 'fp8',
    ):
        self.hardware = hardware
        self.backend = backend
        self.database_mode = database_mode

        if version is None:
            version = get_latest_database_version(
                system=hardware, backend=backend
            )
        self.version = version

        self._db: PerfDatabase = get_database(
            system=hardware, backend=backend, version=version
        )

        # if self._db is None:
        #     self.database_mode = DatabaseMode.SOL
        # Load NCCL database separately (shared across backends)
        self._nccl_db: Optional[PerfDatabase] = None
        # try:
        #     nccl_version = get_latest_database_version(
        #         system=hardware, backend="nccl"
        #     )
        #     self._nccl_db = get_database(
        #         system=hardware, backend="nccl", version=nccl_version
        #     )
        # except Exception:
        #     # NCCL data may not be available for all hardware
        #     pass

        self.gemm_bits = gemm_bits
        self.comm_bits = comm_bits

    @staticmethod
    def list_available() -> dict:
        """List all available hardware/backend/version combinations."""
        return get_supported_databases()

    def get_runtime(self, op: Operator, system: System) -> float:
        """Look up runtime for a given operator.

        Args:
            op: Operator to look up.
            system: System configuration.
        Returns:
            OperatorTiming with latency in seconds.
        """
        op_type = op.get_op_type(op.dim)
        op_dim = op.dim[:op.get_effective_dim_len()]
        if op_type == 'GEMM':
            B, M, N, K = op_dim
            # B, M, N, K = self.dim[:self.get_effective_dim_len()]
            # input_a = (B, K, N)
            # input_w = (M, K)
            # output = (B, M, N)
            latency = float(self.query_gemm(
                m=B*N,
                n=M,
                k=K,
                quant_mode=bits_to_gemm_quants.get(self.gemm_bits, GEMMQuantMode.fp8),
            )) / 1000  # Convert ms → s
            return latency
        if op_type == 'Logit':
            B, H, M, N, D, Hkv = op_dim
            if M > 1:
                return self.query_context_attention(
                    batch_size=B,
                    seq_len=M,
                    num_heads=H,
                    num_kv_heads=Hkv,
                    head_dim=D//Hkv,
                    quant_mode=bits_to_gemm_quants.get(self.gemm_bits, GEMMQuantMode.fp8),
                    kv_cache_dtype=bits_to_gemm_quants.get(self.gemm_bits, KVCacheQuantMode.fp8),
                )/2000  # Logit is roughly half of attention

            return self.query_generation_attention(
                batch_size=B,
                seq_len=M,
                num_heads=H,
                num_kv_heads=Hkv,
                head_dim=D//Hkv,
                kv_cache_dtype=bits_to_gemm_quants.get(self.gemm_bits, KVCacheQuantMode.fp8),
            )/2000
        if op_type == 'Attend':
            B, H, M, N, D, Hkv = op_dim
            if M > 1:
                return self.query_context_attention(
                    batch_size=B,
                    seq_len=M,
                    num_heads=H,
                    num_kv_heads=Hkv,
                    head_dim=D//Hkv,
                    quant_mode=bits_to_gemm_quants.get(self.gemm_bits, GEMMQuantMode.fp8),
                    kv_cache_dtype=bits_to_gemm_quants.get(self.gemm_bits, KVCacheQuantMode.fp8),
                )/2000

            return self.query_generation_attention(
                batch_size=B,
                seq_len=M,
                num_heads=H,
                num_kv_heads=Hkv,
                head_dim=D//Hkv,
                kv_cache_dtype=bits_to_gemm_quants.get(self.gemm_bits, KVCacheQuantMode.fp8),
            )/2000
        if op_type == 'Sync':
            if op.collective_type == CollectiveType.AllReduce:
                return self.query_custom_allreduce(
                    num_gpus=op.num_collective_nodes,
                    message_size_bytes=op.communication_data(),
                    quant_mode=bits_to_comm_quants.get(self.comm_bits, CommQuantMode.fp8),
                )  / 1000
            elif  op.collective_type == CollectiveType.All2All:
                return get_A2A_time(data_size , self.num_collective_nodes, system) / 1000
            elif  op.collective_type == CollectiveType.MessagePass:
                return self._db.query_p2p( message_bytes = op.communication_data(), database_mode=self.database_mode)  / 1000
            elif op.collective_type == CollectiveType.AllGather:
                return self.query_nccl(
                    op_name="all_gather",
                    num_gpus=op.num_collective_nodes,
                    message_size_bytes=op.communication_data(),
                    quant_mode=bits_to_comm_quants.get(self.comm_bits, CommQuantMode.fp8),
                )  / 1000

        return op.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec

    def query_gemm(
        self,
        m: int,
        n: int,
        k: int,
        quant_mode: GEMMQuantMode = GEMMQuantMode.float16,
    ) -> OperatorTiming:
        """Look up GEMM operator runtime.

        Args:
            m: Output rows (typically batch_size * seq_len).
            n: Output columns (typically hidden_size or intermediate_size).
            k: Inner dimension (typically hidden_size).
            quant_mode: Quantization mode for weights/activations.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        result = self._db.query_gemm(
            m=m, n=n, k=k,
            quant_mode=quant_mode,
            database_mode=self.database_mode,
        )
        return self._to_timing(result)

    def query_context_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        quant_mode: FMHAQuantMode = FMHAQuantMode.float16,
        kv_cache_dtype: KVCacheQuantMode = KVCacheQuantMode.float16,
    ) -> OperatorTiming:
        """Look up prefill (context) attention runtime.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Input sequence length.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads (GQA).
            head_dim: Dimension per head.
            quant_mode: Attention quantization mode.
            kv_cache_dtype: KV cache data type.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        result = self._db.query_context_attention(
            b=batch_size,
            s=seq_len,
            prefix=0,
            n=num_heads,
            n_kv=num_kv_heads,
            head_size=head_dim,
            window_size=0,
            fmha_quant_mode=quant_mode,
            kvcache_quant_mode=kv_cache_dtype,
            database_mode=self.database_mode,
        )
        return self._to_timing(result)

    def query_generation_attention(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        kv_cache_dtype: KVCacheQuantMode = KVCacheQuantMode.float16,
    ) -> OperatorTiming:
        """Look up decode (generation) attention runtime.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Current output sequence length.
            num_heads: Number of query attention heads.
            num_kv_heads: Number of key/value heads.
            head_dim: Dimension per head.
            kv_cache_dtype: KV cache data type.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        result = self._db.query_generation_attention(
            b=batch_size,
            s=seq_len,
            n=num_heads,
            n_kv=num_kv_heads,
            head_size=head_dim,
            window_size=0,
            kvcache_quant_mode=kv_cache_dtype,
            database_mode=self.database_mode,
        )
        return self._to_timing(result)

    def query_nccl(
        self,
        op_name: str,
        num_gpus: int,
        message_size_bytes: int,
        quant_mode: CommQuantMode = CommQuantMode.half,
    ) -> OperatorTiming:
        """Look up NCCL collective communication runtime.

        Args:
            op_name: Collective type ('all_reduce', 'all_gather',
                     'reduce_scatter', 'alltoall', 'broadcast').
            num_gpus: Number of GPUs in the topology.
            message_size_bytes: Message size in bytes.
            quant_mode: Communication data type.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        db = self._nccl_db or self._db
        # SDK query_nccl expects element count; convert bytes → elements.
        bytes_per_element = quant_mode.value.memory
        num_elements = message_size_bytes // bytes_per_element
        result = db.query_nccl(
            dtype=quant_mode,
            num_gpus=num_gpus,
            operation=op_name,
            message_size=num_elements,
            database_mode=self.database_mode,
        )
        return self._to_timing(result)

    def query_custom_allreduce(
        self,
        num_gpus: int,
        message_size_bytes: int,
        quant_mode: CommQuantMode = CommQuantMode.half,
    ) -> OperatorTiming:
        """Look up custom (fused) AllReduce runtime.

        Falls back to NCCL all_reduce if custom data is unavailable.
        """
        # SDK query_custom_allreduce expects element count.
        bytes_per_element = quant_mode.value.memory
        num_elements = message_size_bytes // bytes_per_element
        try:
            result = self._db.query_custom_allreduce(
                quant_mode=quant_mode,
                tp_size=num_gpus,
                size=num_elements,
                database_mode=self.database_mode,
            )
            return self._to_timing(result)
        except Exception:
            return self.query_nccl(
                op_name="all_reduce",
                num_gpus=num_gpus,
                message_size_bytes=message_size_bytes,
                quant_mode=quant_mode,
            )

    def query_moe(
        self,
        quant_mode: MoEQuantMode,
        topk: int,
        num_experts: int,
        hidden_size: int,
        inter_size: int,
        moe_tp: int = 1,
        moe_ep: int = 1,
        num_tokens: int = 1,
        workload: str = "power_law_1.01",
    ) -> OperatorTiming:
        """Look up MoE dispatch/compute/combine runtime.

        Args:
            quant_mode: MoE quantization mode.
            topk: Number of experts selected per token.
            num_experts: Total number of experts.
            hidden_size: Model hidden dimension.
            inter_size: FFN intermediate dimension.
            moe_tp: Tensor parallelism for MoE.
            moe_ep: Expert parallelism degree.
            num_tokens: Tokens per forward pass.
            workload: Load distribution. 'power_law_1.01' (near-uniform,
                Mixtral-class) or 'power_law_1.2' (skewed, DeepSeek-class).
                Must match workloads profiled in aiconfigurator.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        # MoE silicon data is sparse for many model configs; use HYBRID
        # which tries silicon first then falls back to analytical.
        # This only affects query_moe — all other queries (GEMM, attention,
        # NCCL) continue using self.database_mode unchanged.
        result = self._db.query_moe(
            quant_mode=quant_mode,
            workload_distribution=workload,
            topk=topk,
            num_experts=num_experts,
            hidden_size=hidden_size,
            inter_size=inter_size,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            num_tokens=num_tokens,
            database_mode=DatabaseMode.HYBRID,
        )
        return self._to_timing(result)

    def query_trtllm_alltoall(
        self,
        op_name: str,
        num_tokens: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: MoEQuantMode = MoEQuantMode.float16,
    ) -> OperatorTiming:
        """Look up TRT-LLM All2All communication latency.

        Args:
            op_name: One of 'alltoall_dispatch', 'alltoall_combine',
                     'alltoall_prepare'.
            num_tokens: Tokens being communicated.
            hidden_size: Model hidden dimension.
            topk: Experts activated per token.
            num_experts: Total experts.
            moe_ep_size: Expert parallelism degree.
            quant_mode: MoE quantization mode.

        Returns:
            OperatorTiming with latency in microseconds.
        """
        result = self._db.query_trtllm_alltoall(
            op_name=op_name,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
            quant_mode=quant_mode,
            database_mode=self.database_mode,
        )
        return self._to_timing(result)

    def _to_timing(self, result) -> OperatorTiming:
        """Convert aiconfigurator PerformanceResult to OperatorTiming.

        Aiconfigurator returns latency in milliseconds; we convert to
        microseconds for the simulator.
        """
        latency_ms = float(result)
        energy = getattr(result, "energy", None)
        source = (
            "silicon"
            if self.database_mode == DatabaseMode.SILICON
            else self.database_mode.name.lower()
        )
        return latency_ms

    def __repr__(self) -> str:
        return (
            f"RuntimeDB(hardware={self.hardware!r}, "
            f"backend={self.backend!r}, "
            f"version={self.version!r}, "
            f"mode={self.database_mode.name})"
        )
