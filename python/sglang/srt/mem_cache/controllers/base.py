"""Backend-neutral cache-controller contract.

``UnifiedRadixCache`` owns the radix-tree topology and scheduling policy.  A
controller owns the movement of cache payloads between the locations managed by
one cache backend.  HiCache implements this contract with a local host pool and
an optional L3 backend; LMCache can implement it without inheriting HiCache's
host-pool and storage-thread machinery.

Request retraction is exposed as a high-level synchronous operation so Unified
does not depend on a backend's allocator, transfer engine, or private L2
transfer builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.common import RetractionBackup
    from sglang.srt.mem_cache.hicache_storage import PoolTransfer
    from sglang.srt.mem_cache.memory_pool_host import PoolEntry


@dataclass(frozen=True)
class KVCacheGroupRegistration:
    """One device-KV address space exported to an external controller."""

    group_id: int
    layer_indices: tuple[int, ...]
    device_tensors: tuple[torch.Tensor, ...]
    tokens_per_block: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if not self.layer_indices:
            raise ValueError("KV cache registration requires layer_indices")
        if not self.device_tensors:
            raise ValueError("KV cache registration requires device_tensors")
        if self.tokens_per_block <= 0:
            raise ValueError("tokens_per_block must be positive")


@dataclass(frozen=True)
class KVCacheRegistration:
    """Per-rank GPU KV registration for an external controller.

    An MP implementation can export CUDA IPC memory handles for these tensors
    once, then use group-local block IDs in request-level load/store messages.
    """

    instance_id: int
    model_name: str
    world_size: int
    worker_id: int
    groups: tuple[KVCacheGroupRegistration, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if not self.groups:
            raise ValueError("KV cache registration requires at least one group")


class BaseController(ABC):
    """Runtime contract between ``UnifiedRadixCache`` and a cache backend.

    Implementations own backend-specific memory and I/O.  In particular, an
    implementation is not required to derive from ``HiCacheController`` or to
    let HiCache allocate its memory.
    """

    def register_kv_cache(self, registration: KVCacheRegistration) -> None:
        """Register device KV storage with an external controller.

        The in-process default is a no-op.  An out-of-process controller can use
        it to exchange CUDA IPC handles before request-level operations.
        """

    def unregister_kv_cache(self) -> None:
        """Release state created by :meth:`register_kv_cache`."""

    @abstractmethod
    def reset(self) -> None:
        """Reset queues and backend-owned transient state."""

    def register_host_pool_entry(self, entry: PoolEntry) -> None:
        """Register an optional Unified sidecar pool with the controller."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support additional host pools"
        )

    @abstractmethod
    def clear_storage_backend(self) -> bool:
        """Clear persistent entries owned by the controller's storage backend."""

    def clear_host_cache(self) -> None:
        """Clear controller-owned local host-cache allocations, if any."""

    def host_cache_available_size(self, pool_name: Any = None) -> int:
        """Return free slots in a controller-owned local host pool."""
        raise NotImplementedError(
            f"{type(self).__name__} does not own a local host cache"
        )

    def alloc_host_cache(
        self, num_slots: int, pool_name: Any = None
    ) -> Optional[torch.Tensor]:
        """Allocate slots from a controller-owned local host pool."""
        raise NotImplementedError(
            f"{type(self).__name__} does not own a local host cache"
        )

    def free_host_cache(
        self, indices: torch.Tensor, pool_name: Any = None
    ) -> int:
        """Release slots in a controller-owned local host pool."""
        raise NotImplementedError(
            f"{type(self).__name__} does not own a local host cache"
        )

    def backup_retraction(
        self,
        device_indices: torch.Tensor,
        extra_pools: Optional[list[PoolTransfer]] = None,
        *,
        reclaim_host: Optional[Callable[[int], int]] = None,
        request_id: Optional[str] = None,
    ) -> Optional[RetractionBackup]:
        """Synchronously preserve request KV before its GPU slots are released."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support retraction backup"
        )

    def restore_retraction(
        self,
        backup: RetractionBackup,
        device_indices: torch.Tensor,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> None:
        """Synchronously restore a request into caller-provided GPU slots."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support retraction restore"
        )

    def discard_retraction(self, backup: RetractionBackup) -> None:
        """Release resources held by an unused retraction backup."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support retraction backup"
        )

    def get_storage_stats(self) -> Any:
        """Return backend storage statistics without exposing the backend object."""
        return None

    @abstractmethod
    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        """Submit a device-to-backend write and return its backend indices."""

    @abstractmethod
    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        """Submit a backend-to-device load and return allocated device slots."""

    @abstractmethod
    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        """Submit a persistent-store operation and return its operation ID."""

    @abstractmethod
    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Any:
        """Start a persistent-cache lookup and prefetch operation."""

    @abstractmethod
    def terminate_prefetch(self, operation: Any) -> tuple[int, list[str]]:
        """Terminate a prefetch and return the completed prefix metadata."""

    @abstractmethod
    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> None:
        """Schedule release of backend memory associated with a transfer."""

    @abstractmethod
    def prefetch_rate_limited(self) -> bool:
        """Return whether another prefetch should currently be rejected."""

    @abstractmethod
    def start_loading(self) -> int:
        """Submit queued loads and return their layer-event producer ID."""

    @abstractmethod
    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Return the attention CP rank and world size used for metrics."""
