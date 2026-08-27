"""Unified radix tree backed by the LMCache multiprocess service.

The class reuses ``UnifiedRadixCache`` only for its device radix tree and
component machinery.  It does not initialize HiCache, does not allocate a
host pool, and does not expose LMCache's internal CPU/remote hierarchy to the
tree.  External hits are loaded into private GPU slots before they are
published into the radix tree.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.connectors.lmcache import (
    LMCacheLoadOperation,
    LMCacheLookupOperation,
    LMCacheMPConnector,
    LMCacheStoreOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import NodeId, UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


@dataclass
class _ExternalFlow:
    key: RadixKey
    initial_local_hit: int
    lookup: LMCacheLookupOperation
    load: Optional[LMCacheLoadOperation] = None
    anchor_node: Optional[NodeId] = None
    anchor_lock: Optional[DecLockRefParams] = None
    cancelled: bool = False


@dataclass
class _PendingStore:
    operation: LMCacheStoreOperation
    node_id: NodeId
    lock_params: DecLockRefParams


class LMCacheUnifiedRadixCache(UnifiedRadixCache):
    """FULL-attention Unified radix tree with device-direct LMCache MP I/O."""

    def __init__(
        self,
        params: CacheInitParams,
        *,
        model_config: ModelConfig,
        tp_size: int,
        tp_rank: int,
        lmcache_config_file: Optional[str],
    ) -> None:
        if tuple(params.tree_components or ()) != (ComponentType.FULL,):
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache currently supports FULL-attention "
                "Unified trees only"
            )
        if params.pp_size != 1:
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache does not support pipeline parallelism"
            )

        super().__init__(params)
        kv_tensors, is_mla = self._resolve_registered_tensors()
        self.lmcache_connector = LMCacheMPConnector(
            config_file=lmcache_config_file,
            model_name=model_config.model_path,
            world_size=tp_size,
            worker_id=tp_rank,
            tp_group=params.tp_cache_group,
            page_size=self.page_size,
            kv_tensors=kv_tensors,
            is_mla=is_mla,
        )
        self._external_flows: dict[str, _ExternalFlow] = {}
        self._pending_stores: list[_PendingStore] = []
        self._finished_store_requests: set[str] = set()
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self._lmcache_closed = False
        # Kept only for compatibility with the scheduler's existing HiCache
        # prefetch entry point. LMCache does not pass hash-prefix keys.
        self.hicache_storage_pass_prefix_keys = False
        atexit.register(self.shutdown)

    def _resolve_registered_tensors(self) -> tuple[list[torch.Tensor], bool]:
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        if getattr(kv_pool, "use_dsa", False):
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache does not yet register DSA indexer pools"
            )
        kv_buffer = getattr(kv_pool, "kv_buffer", None)
        if kv_buffer is not None:
            return list(kv_buffer), True

        k_buffer = getattr(kv_pool, "k_buffer", None)
        v_buffer = getattr(kv_pool, "v_buffer", None)
        if not k_buffer or not v_buffer or len(k_buffer) != len(v_buffer):
            raise NotImplementedError(
                f"Unsupported SGLang KV pool for LMCache MP: {type(kv_pool).__name__}"
            )
        if getattr(kv_pool, "kv_cache_layout", "nhd") != "nhd":
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache currently supports NHD MHA pools only"
            )
        return [*k_buffer, *v_buffer], False

    # ------------------------------------------------------------------
    # Lookup + asynchronous retrieve
    # ------------------------------------------------------------------

    @staticmethod
    def _external_cache_salt(
        cache_salt: Optional[str], extra_key: Optional[str]
    ) -> str:
        """Namespace remote entries by both SGLang key dimensions.

        LMCache has one cache-salt field while SGLang's radix key keeps the
        user salt and multimodal/adapter extra key separately. Hashing both
        avoids external hits crossing those namespaces.
        """
        if not extra_key:
            return cache_salt or ""
        payload = f"{cache_salt or ''}\0{extra_key}".encode()
        return hashlib.sha256(payload).hexdigest()

    def is_backuped(self, node_id: NodeId) -> bool:
        # LMCache is external to the tree. Any resident L1 boundary is a valid
        # lookup anchor because lookup keys are rebuilt from the complete token
        # sequence rather than from a host-tree hash chain.
        return True

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        request_extra_key: Optional[str] = None,
        request_cache_salt: Optional[str] = None,
    ) -> None:
        del last_hash, prefix_keys
        if req_id in self._external_flows:
            return
        local_tokens = list(matched_prefix_tokens or [])
        token_ids = local_tokens + list(new_input_tokens)
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        extra_key = request_extra_key or anchor_extra_key
        cache_salt = request_cache_salt or anchor_cache_salt
        key = RadixKey(
            token_ids,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(key) == 0:
            return
        token_ids = key.raw_token_ids()[: len(key)]
        lookup = self.lmcache_connector.submit_lookup(
            req_id,
            token_ids,
            local_hit_tokens=min(len(local_tokens), len(key)),
            cache_salt=self._external_cache_salt(cache_salt, extra_key),
        )
        self._external_flows[req_id] = _ExternalFlow(
            key=key,
            initial_local_hit=min(len(local_tokens), len(key)),
            lookup=lookup,
        )

    def _allocate_external_slots(self, num_tokens: int) -> Optional[torch.Tensor]:
        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            self.evict(EvictParams(num_tokens=num_tokens))
        return self.token_to_kv_pool_allocator.alloc(num_tokens)

    def _start_external_load(self, flow: _ExternalFlow, total_hit: int) -> bool:
        latest = super().match_prefix(MatchPrefixParams(key=flow.key))
        local_hit = len(latest.device_indices)
        total_hit = min(total_hit, len(flow.key))
        if total_hit <= local_hit:
            self.prefetch_loaded_tokens_by_reqid[flow.lookup.request_id] = 0
            return False

        # Pin the exact L1 boundary before allocation: allocator eviction may
        # run below, and the latest match can be deeper than the request's
        # original (already pinned) prefix.
        flow.anchor_node = latest.last_device_node
        flow.anchor_lock = self.inc_lock_ref(flow.anchor_node).to_dec_params()
        num_tokens = total_hit - local_hit
        device_indices = self._allocate_external_slots(num_tokens)
        allocation_ok = torch.tensor(
            [int(device_indices is not None)], dtype=torch.int32, device="cpu"
        )
        self._all_reduce(allocation_ok, torch.distributed.ReduceOp.MIN)
        if not allocation_ok.item():
            if device_indices is not None:
                self.token_to_kv_pool_allocator.free(device_indices)
            self._release_flow_anchor(flow)
            logger.debug(
                "LMCache retrieve declined for %s: a TP rank cannot allocate "
                "%d GPU slots",
                flow.lookup.request_id,
                num_tokens,
            )
            return False
        assert device_indices is not None

        try:
            flow.load = self.lmcache_connector.submit_load(
                flow.lookup,
                device_indices,
                local_hit_tokens=local_hit,
            )
        except Exception:
            self.token_to_kv_pool_allocator.free(device_indices)
            self.dec_lock_ref(flow.anchor_node, flow.anchor_lock)
            flow.anchor_node = None
            flow.anchor_lock = None
            logger.exception(
                "LMCache retrieve submission failed for %s", flow.lookup.request_id
            )
            return False
        return True

    def check_prefetch_progress(self, req_id: str) -> bool:
        flow = self._external_flows.get(req_id)
        if flow is None:
            return True
        if flow.cancelled:
            return False

        if flow.load is None:
            total_hit = self.lmcache_connector.poll_lookup(flow.lookup)
            if total_hit is None:
                return False
            if total_hit <= flow.initial_local_hit or not self._start_external_load(
                flow, total_hit
            ):
                self.lmcache_connector.end_lookup(req_id)
                self._external_flows.pop(req_id, None)
                self.prefetch_loaded_tokens_by_reqid.setdefault(req_id, 0)
                return True
            return False

        if flow.load.result is None:
            return False
        if not flow.load.result:
            self._finish_failed_load(flow)
            return True

        self._publish_loaded_prefix(flow)
        return True

    def _release_flow_anchor(self, flow: _ExternalFlow) -> None:
        if flow.anchor_node is not None and flow.anchor_lock is not None:
            self.dec_lock_ref(flow.anchor_node, flow.anchor_lock)
        flow.anchor_node = None
        flow.anchor_lock = None

    def _finish_failed_load(self, flow: _ExternalFlow) -> None:
        assert flow.load is not None
        self.token_to_kv_pool_allocator.free(flow.load.device_indices)
        self._release_flow_anchor(flow)
        rid = flow.lookup.request_id
        self.lmcache_connector.end_lookup(rid)
        self._external_flows.pop(rid, None)
        if flow.cancelled:
            self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        else:
            self.prefetch_loaded_tokens_by_reqid[rid] = 0
        if rid in self._finished_store_requests:
            self._finish_store_session_if_idle(rid)

    def _publish_loaded_prefix(self, flow: _ExternalFlow) -> None:
        assert flow.load is not None and flow.load.result
        rid = flow.lookup.request_id
        total_hit = flow.load.end
        latest = super().match_prefix(MatchPrefixParams(key=flow.key))
        local_hit = len(latest.device_indices)
        skip = max(local_hit - flow.load.local_hit_tokens, 0)
        if skip:
            self.token_to_kv_pool_allocator.free(flow.load.device_indices[:skip])
        suffix = flow.load.device_indices[skip : skip + max(total_hit - local_hit, 0)]

        if local_hit < total_hit:
            combined = torch.cat([latest.device_indices, suffix])
            self.insert(
                InsertParams(
                    key=flow.key[:total_hit],
                    value=combined,
                    prev_prefix_len=local_hit,
                )
            )
        elif suffix.numel():
            self.token_to_kv_pool_allocator.free(suffix)

        self.prefetch_loaded_tokens_by_reqid[rid] = max(total_hit - local_hit, 0)
        self._release_flow_anchor(flow)
        self.lmcache_connector.end_lookup(rid)
        self._external_flows.pop(rid, None)

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    # ------------------------------------------------------------------
    # Asynchronous store
    # ------------------------------------------------------------------

    def _submit_store(self, req: Req, token_ids: list[int]) -> None:
        key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        if len(key) == 0:
            return
        matched = super().match_prefix(MatchPrefixParams(key=key))
        if len(matched.device_indices) < len(key):
            logger.warning(
                "LMCache store skipped for %s: radix prefix has %d/%d tokens",
                req.rid,
                len(matched.device_indices),
                len(key),
            )
            return
        lock_params = self.inc_lock_ref(matched.last_device_node).to_dec_params()
        try:
            operation = self.lmcache_connector.submit_store(
                req.rid,
                key.raw_token_ids()[: len(key)],
                matched.device_indices[: len(key)],
                cache_salt=self._external_cache_salt(
                    req.cache_salt, req.extra_key
                ),
            )
        except Exception:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            logger.exception("LMCache store submission failed for %s", req.rid)
            return
        if operation is None:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            return
        self._pending_stores.append(
            _PendingStore(operation, matched.last_device_node, lock_params)
        )

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        super().cache_unfinished_req(req, chunked=chunked, **kwargs)
        self._submit_store(req, req.get_fill_ids())

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if not is_insert:
            self.release_aborted_request(req.rid)
        super().cache_finished_req(
            req,
            is_insert=is_insert,
            kv_len_to_handle=kv_len_to_handle,
            **kwargs,
        )
        if is_insert:
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
            self._submit_store(req, token_ids)
        self._finished_store_requests.add(req.rid)
        self._finish_store_session_if_idle(req.rid)

    def _finish_store_session_if_idle(self, rid: str) -> None:
        if rid not in self._finished_store_requests:
            return
        if self._has_pending_store(rid) or rid in self._external_flows:
            return
        self._finished_store_requests.discard(rid)
        self.lmcache_connector.finish_request(rid)

    def _has_pending_store(self, rid: str) -> bool:
        return any(p.operation.request_id == rid for p in self._pending_stores)

    # ------------------------------------------------------------------
    # Future polling and lifecycle
    # ------------------------------------------------------------------

    def _ready_prefix_count(self, operations) -> int:
        count = 0
        for operation in operations:
            if not operation.query():
                break
            count += 1
        tensor = torch.tensor([count], dtype=torch.int64, device="cpu")
        self._all_reduce(tensor, torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def check_hicache_events(self) -> None:
        """Poll LMCache retrieve/store futures at the scheduler safe point."""
        load_flows = [
            flow
            for flow in self._external_flows.values()
            if flow.load is not None and flow.load.result is None
        ]
        ready_loads = self._ready_prefix_count([f.load for f in load_flows])
        for flow in load_flows[:ready_loads]:
            assert flow.load is not None
            self.lmcache_connector.complete_load(flow.load)
            if flow.cancelled:
                self._finish_failed_load(flow)

        ready_stores = self._ready_prefix_count(
            [pending.operation for pending in self._pending_stores]
        )
        for _ in range(ready_stores):
            pending = self._pending_stores.pop(0)
            self.lmcache_connector.complete_store(pending.operation)
            self.dec_lock_ref(pending.node_id, pending.lock_params)
            self._finish_store_session_if_idle(pending.operation.request_id)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._finished_store_requests.add(rid)
        flow = self._external_flows.get(rid)
        if flow is None:
            self.lmcache_connector.end_lookup(rid)
            self._finish_store_session_if_idle(rid)
            return
        flow.cancelled = True
        if flow.load is None:
            self.lmcache_connector.end_lookup(rid)
            self._external_flows.pop(rid, None)
            self._finish_store_session_if_idle(rid)
        elif flow.load.result is not None:
            self._finish_failed_load(flow)

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> tuple[torch.Tensor, NodeId]:
        # LMCache retrieves are completed before the request is admitted, so
        # the scheduler should never observe an external hit as a host hit.
        return self.tree_core.empty_match_result.device_indices, params.best_match_node

    def ready_to_load_host_cache(self) -> int:
        return -1

    def supports_retraction_backup(self) -> bool:
        return False

    def clear_storage_backend(self) -> bool:
        return self.lmcache_connector.clear()

    def reset(self) -> None:
        # ``UnifiedRadixCache.__init__`` calls virtual reset before the connector
        # exists, hence all LMCache cleanup is presence-gated.
        connector = getattr(self, "lmcache_connector", None)
        if connector is not None:
            for flow in list(getattr(self, "_external_flows", {}).values()):
                if flow.load is not None:
                    try:
                        flow.load.future.result(timeout=connector.operation_timeout)
                    except Exception:
                        pass
                    connector.complete_load(flow.load, synchronize=False)
                    self.token_to_kv_pool_allocator.free(flow.load.device_indices)
                    self._release_flow_anchor(flow)
                connector.end_lookup(flow.lookup.request_id)
            for pending in list(getattr(self, "_pending_stores", [])):
                try:
                    pending.operation.future.result(
                        timeout=connector.operation_timeout
                    )
                except Exception:
                    pass
                connector.complete_store(pending.operation, synchronize=False)
                self.dec_lock_ref(pending.node_id, pending.lock_params)
            self._external_flows.clear()
            self._pending_stores.clear()
            self._finished_store_requests.clear()
            self.prefetch_loaded_tokens_by_reqid.clear()
            connector.end_all_sessions()
        super().reset()

    def shutdown(self) -> None:
        if getattr(self, "_lmcache_closed", True):
            return
        self.reset()
        self.lmcache_connector.close()
        self._lmcache_closed = True
