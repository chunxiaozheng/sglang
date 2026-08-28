"""LMCache multiprocess connector used by ``LMCacheUnifiedRadixCache``.

This module deliberately talks to LMCache's engine-neutral MP protocol.  It
does not use LMCache's legacy SGLang integration and it never constructs an
in-process LMCache engine.  The registered SGLang GPU KV tensors remain owned
by SGLang; LMCache accesses them through device-memory and event IPC handles.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_DEFAULT_MQ_TIMEOUT_SECONDS = 300.0
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


class _ImmediateFuture:
    """Minimal future used to preserve TP operation order on local failure."""

    def __init__(self, result: bool) -> None:
        self._result = result

    def query(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> bool:
        del timeout
        return self._result

    def retain_reference(self, value: object) -> None:
        del value


@dataclass
class LMCacheLookupOperation:
    request_id: str
    token_ids: list[int]
    local_hit_tokens: int
    cache_salt: str
    submission_future: Any = None
    completion_future: Any = None
    total_hit_tokens: Optional[int] = None
    locks_held: bool = False
    lock_start: int = 0


@dataclass
class LMCacheLoadOperation:
    request_id: str
    token_ids: list[int]
    start: int
    end: int
    local_hit_tokens: int
    device_indices: torch.Tensor
    future: Any
    lookup: LMCacheLookupOperation
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


@dataclass
class LMCacheStoreOperation:
    request_id: str
    start: int
    end: int
    future: Any
    result: Optional[bool] = None

    def query(self) -> bool:
        return self.result is not None or bool(self.future.query())


class LMCacheMPConnector:
    """Asynchronous, CUDA-IPC connector to a standalone LMCache server."""

    def __init__(
        self,
        *,
        config_file: Optional[str],
        model_name: str,
        world_size: int,
        worker_id: int,
        tp_group: Optional[dist.ProcessGroup],
        page_size: int,
        kv_tensors: list[torch.Tensor],
        is_mla: bool,
    ) -> None:
        try:
            import zmq
            from lmcache.v1.config import load_engine_config_with_overrides
            from lmcache.v1.multiprocess.mq import MessageQueueClient
        except ImportError as exc:
            raise ImportError(
                "LMCacheUnifiedRadixCache requires the `lmcache` package and "
                "a running LMCache multiprocess server."
            ) from exc

        if not kv_tensors:
            raise ValueError("LMCache KV tensor registration cannot be empty")
        if any(t.device.type != "cuda" for t in kv_tensors):
            raise NotImplementedError("LMCache MP currently requires CUDA KV tensors")
        if any(t.device != kv_tensors[0].device for t in kv_tensors):
            raise ValueError("All LMCache-registered KV tensors must share one device")
        if any(t.dim() != 3 for t in kv_tensors):
            raise NotImplementedError(
                "LMCache MP currently supports SGLang NHD/MLA 3-D KV tensors only"
            )
        if not is_mla and kv_tensors[0].shape[1] == 1:
            # LMCache's SGLang wire-format detector distinguishes fused MLA
            # from split MHA using the middle dimension. A one-local-head MHA
            # tensor is byte-compatible with (2, head_dim / 2); register that
            # zero-copy view so common GQA+TP configurations remain MHA. The
            # transfer kernel only needs the per-token contiguous byte span.
            if any(t.shape[1] != 1 or t.shape[2] % 2 for t in kv_tensors):
                raise NotImplementedError(
                    "One-head SGLang MHA requires an even per-head dimension "
                    "for LMCache MP format disambiguation"
                )
            kv_tensors = [
                tensor.view(tensor.shape[0], 2, tensor.shape[2] // 2)
                for tensor in kv_tensors
            ]

        config = load_engine_config_with_overrides(config_file_path=config_file)
        if not config.mp_host:
            raise ValueError(
                "LMCache MP config must define mp_host; pass "
                "--lmcache-config-file or LMCACHE_CONFIG_FILE"
            )
        host = str(config.mp_host)
        if "://" not in host:
            host = f"tcp://{host}"
        self.server_url = f"{host.rstrip(':')}:{int(config.mp_port)}"
        self._mq_timeout = float(
            config.get_extra_config_value(
                "lmcache.mp.mq_timeout", _DEFAULT_MQ_TIMEOUT_SECONDS
            )
        )
        self._heartbeat_interval = float(
            config.get_extra_config_value(
                "lmcache.mp.heartbeat_interval",
                _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            )
        )

        self.model_name = model_name
        self.world_size = int(world_size)
        self.worker_id = int(worker_id)
        self.tp_group = tp_group
        if self.world_size > 1:
            if self.tp_group is None:
                raise ValueError("LMCache TP>1 requires a CPU TP process group")
            group_size = dist.get_world_size(group=self.tp_group)
            if group_size != self.world_size:
                raise NotImplementedError(
                    "LMCache MP requires its synchronization group to match "
                    f"the registered world size, got {group_size=} and "
                    f"{self.world_size=}"
                )
            self._lookup_leader = dist.get_rank(group=self.tp_group) == 0
        else:
            self._lookup_leader = True
        self.page_size = int(page_size)
        self.device = kv_tensors[0].device
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self._kv_caches = {f"kv_{i}": tensor for i, tensor in enumerate(kv_tensors)}
        self._context = zmq.Context.instance()
        self._mq_client = MessageQueueClient(self.server_url, self._context)
        self._transfer_ctx: Any = None
        self._event_backend: Any = None
        self._registered = False
        self._closed = False
        self._lookups: dict[str, LMCacheLookupOperation] = {}
        self._active_sessions: set[str] = set()
        self._store_submitted_tokens: dict[str, int] = {}
        self._control_futures: list[Any] = []
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        self.chunk_size = self._get_chunk_size()
        if self.chunk_size <= 0 or self.chunk_size % self.page_size:
            raise ValueError(
                f"LMCache chunk size {self.chunk_size} must be a positive "
                f"multiple of SGLang page size {self.page_size}"
            )
        self.blocks_in_chunk = self.chunk_size // self.page_size
        self.register_kv_cache()
        self._start_heartbeat()

    @staticmethod
    def _send_request(mq_client: Any, request_type: Any, payloads: list[Any]):
        from lmcache.v1.multiprocess.protocol import get_response_class

        return mq_client.submit_request(
            request_type, payloads, get_response_class(request_type)
        )

    def _get_chunk_size(self) -> int:
        from lmcache.v1.multiprocess.protocol import RequestType

        return int(
            self._send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, []).result(
                timeout=self._mq_timeout
            )
        )

    @property
    def is_lookup_leader(self) -> bool:
        return self._lookup_leader

    @property
    def operation_timeout(self) -> float:
        return self._mq_timeout

    def _sync_leader_int(self, value: int) -> int:
        if self.world_size == 1:
            return value
        tensor = torch.tensor([value], dtype=torch.int64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.tp_group)
        return int(tensor.item())

    def _sync_success(self, success: bool) -> bool:
        if self.world_size == 1:
            return success
        tensor = torch.tensor([int(success)], dtype=torch.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.tp_group)
        return bool(tensor.item())

    def register_kv_cache(self) -> None:
        """Export the SGLang GPU tensors to the LMCache MP server once."""
        from lmcache.utils import EngineType
        from lmcache.v1.multiprocess.transfer_context import create_transfer_context
        from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

        if self._registered:
            raise RuntimeError("LMCache KV tensors are already registered")
        self._event_backend = get_event_ipc_backend(self.device)
        self._event_backend.check_event_support(self.device)
        self._transfer_ctx = create_transfer_context(
            self._kv_caches, mode="lmcache_driven"
        )
        try:
            self._transfer_ctx.register(
                self.instance_id,
                self._kv_caches,
                self.model_name,
                self.world_size,
                self.blocks_in_chunk,
                self._mq_client,
                self._mq_timeout,
                self._send_request,
                layout_hints={"tokens_per_block": self.page_size},
                engine_group_infos=(),
                engine_type=EngineType.SGLANG,
            )
        except Exception:
            self._transfer_ctx.close()
            self._transfer_ctx = None
            raise
        self._registered = True

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def heartbeat() -> None:
            from lmcache.v1.multiprocess.protocol import RequestType

            while not self._heartbeat_stop.wait(self._heartbeat_interval):
                try:
                    self._send_request(
                        self._mq_client, RequestType.PING, [self.instance_id]
                    ).result(timeout=self._heartbeat_interval)
                except Exception:
                    logger.warning("LMCache MP heartbeat failed", exc_info=True)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat, name="sglang-lmcache-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _new_event(self) -> Any:
        event = self._event_backend.create_event(self.device)
        stream = torch.get_device_module(self.device).current_stream()
        self._event_backend.record_event(event, stream)
        return event

    def _create_key(
        self,
        operation: LMCacheLookupOperation,
        *,
        start: int,
        end: int,
        worker_id: Optional[int],
    ) -> Any:
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=worker_id,
            # TODO(chunxiaozheng): optimie MLA
            # Each TP rank registers and retrieves its own worker_id-scoped
            # object, so every stored KV object has exactly one reader.
            num_kv_readers=1,
            token_ids=tuple(operation.token_ids),
            start=start,
            end=end,
            request_id=operation.request_id,
            cache_salt=operation.cache_salt,
        )

    def submit_lookup(
        self,
        request_id: str,
        token_ids: list[int],
        *,
        local_hit_tokens: int,
        cache_salt: str,
    ) -> LMCacheLookupOperation:
        from lmcache.v1.multiprocess.protocol import RequestType

        # A chunked-prefill request can look up the same request_id again.
        # Retire only the preceding lookup; the server session also owns any
        # in-flight stores and must live until finish_request().
        self.end_lookup(request_id)
        operation = LMCacheLookupOperation(
            request_id=request_id,
            token_ids=list(token_ids),
            local_hit_tokens=int(local_hit_tokens),
            cache_salt=cache_salt,
        )
        self._lookups[request_id] = operation
        self._active_sessions.add(request_id)
        aligned_end = len(token_ids) // self.chunk_size * self.chunk_size
        if aligned_end == 0:
            operation.total_hit_tokens = 0
            return operation

        submitted = True
        if self.is_lookup_leader:
            try:
                key = self._create_key(
                    operation, start=0, end=aligned_end, worker_id=None
                )
                operation.submission_future = self._send_request(
                    self._mq_client,
                    RequestType.LOOKUP,
                    [key, self.world_size],
                )
            except Exception:
                logger.exception("LMCache lookup submission failed for %s", request_id)
                submitted = False
        if not self._sync_success(submitted):
            operation.total_hit_tokens = 0
        return operation

    def poll_lookup(self, operation: LMCacheLookupOperation) -> Optional[int]:
        """Return total hit tokens, or ``None`` while lookup+prefetch runs."""
        from lmcache.v1.multiprocess.protocol import RequestType

        if operation.total_hit_tokens is not None:
            return operation.total_hit_tokens
        result = -1
        if self.is_lookup_leader:
            try:
                if operation.submission_future is None:
                    result = 0
                elif not operation.submission_future.query():
                    result = -1
                elif operation.completion_future is None:
                    operation.submission_future.result(timeout=0)
                    operation.completion_future = self._send_request(
                        self._mq_client,
                        RequestType.WAIT_PREFETCH_STATUS,
                        [operation.request_id, self._mq_timeout],
                    )
                    result = -1
                elif not operation.completion_future.query():
                    result = -1
                else:
                    matched_chunks = operation.completion_future.result(timeout=0)
                    result = (
                        0
                        if matched_chunks is None
                        else int(matched_chunks) * self.chunk_size
                    )
            except Exception:
                logger.exception(
                    "LMCache lookup completion failed for %s", operation.request_id
                )
                result = 0

        result = self._sync_leader_int(result)
        if result < 0:
            return None
        aligned_end = len(operation.token_ids) // self.chunk_size * self.chunk_size
        operation.total_hit_tokens = min(max(result, 0), aligned_end)
        operation.locks_held = operation.total_hit_tokens > 0
        return operation.total_hit_tokens

    def _slots_to_blocks(self, slots: torch.Tensor) -> list[int]:
        if slots.numel() == 0:
            return []
        if slots.numel() % self.page_size:
            raise ValueError("LMCache slots must contain complete SGLang pages")
        pages = slots.detach().to(dtype=torch.int64, device="cpu").reshape(
            -1, self.page_size
        )
        starts = pages[:, 0]
        expected = starts[:, None] + torch.arange(self.page_size, dtype=torch.int64)
        if torch.any(starts % self.page_size) or not torch.equal(pages, expected):
            raise ValueError("LMCache slots must be page-aligned contiguous pages")
        return (starts // self.page_size).tolist()

    def _free_lookup_locks(
        self, operation: LMCacheLookupOperation, start: int, end: int
    ) -> None:
        if not self.is_lookup_leader or start >= end:
            return
        from lmcache.v1.multiprocess.protocol import RequestType

        key = self._create_key(operation, start=start, end=end, worker_id=None)
        self._track_control_future(
            self._send_request(
                self._mq_client,
                RequestType.FREE_LOOKUP_LOCKS,
                [key, self.world_size],
            )
        )

    def _track_control_future(self, future: Any) -> None:
        # Keep fire-and-forget control RPCs alive and bound the local list.
        self._control_futures = [f for f in self._control_futures if not f.query()]
        self._control_futures.append(future)

    def _flush_control_futures(self) -> None:
        for future in self._control_futures:
            try:
                future.result(timeout=self._mq_timeout)
            except Exception:
                logger.warning(
                    "LMCache control RPC failed during shutdown", exc_info=True
                )
        self._control_futures.clear()

    def submit_load(
        self,
        operation: LMCacheLookupOperation,
        device_indices: torch.Tensor,
        *,
        local_hit_tokens: int,
    ) -> LMCacheLoadOperation:
        if operation.total_hit_tokens is None or not operation.locks_held:
            raise RuntimeError("LMCache load requires a completed, locked lookup")
        total_hit = operation.total_hit_tokens
        start = local_hit_tokens // self.chunk_size * self.chunk_size
        prefix_pad = local_hit_tokens - start
        fresh_blocks = self._slots_to_blocks(device_indices)
        prefix_pages = prefix_pad // self.page_size
        block_ids = [0] * prefix_pages + fresh_blocks
        key = self._create_key(
            operation, start=start, end=total_hit, worker_id=self.worker_id
        )
        self._free_lookup_locks(operation, 0, start)
        operation.lock_start = start
        try:
            event = self._new_event()
            future = self._transfer_ctx.submit_retrieve(
                operation.request_id,
                key,
                self.instance_id,
                self._kv_caches,
                [block_ids],
                event,
                self.blocks_in_chunk,
                skip_first_n_tokens=prefix_pad,
            )
        except Exception:
            # Every TP rank must enqueue one operation in the same order. A
            # ready-false future lets completion consensus fail the operation
            # after successful peers finish their already-submitted retrieve.
            logger.exception(
                "LMCache retrieve submission failed for %s",
                operation.request_id,
            )
            future = _ImmediateFuture(False)
            event = None
        if event is not None:
            future.retain_reference(event)
        return LMCacheLoadOperation(
            request_id=operation.request_id,
            token_ids=operation.token_ids,
            start=start,
            end=total_hit,
            local_hit_tokens=local_hit_tokens,
            device_indices=device_indices,
            future=future,
            lookup=operation,
        )

    def complete_load(
        self, operation: LMCacheLoadOperation, *, synchronize: bool = True
    ) -> bool:
        if operation.result is not None:
            return operation.result
        success = False
        try:
            success = bool(operation.future.result(timeout=0))
        except Exception:
            logger.exception("LMCache retrieve failed for %s", operation.request_id)
        if synchronize:
            success = self._sync_success(success)
        if not success and operation.lookup.locks_held:
            self._free_lookup_locks(
                operation.lookup, operation.start, operation.end
            )
        operation.lookup.locks_held = False
        operation.result = success
        return success

    def submit_store(
        self,
        request_id: str,
        token_ids: list[int],
        device_indices: torch.Tensor,
        *,
        cache_salt: str,
    ) -> Optional[LMCacheStoreOperation]:
        aligned_end = len(token_ids) // self.chunk_size * self.chunk_size
        start = min(self._store_submitted_tokens.get(request_id, 0), aligned_end)
        start = start // self.chunk_size * self.chunk_size
        if aligned_end <= start:
            return None
        self._active_sessions.add(request_id)
        lookup = LMCacheLookupOperation(
            request_id=request_id,
            token_ids=list(token_ids),
            local_hit_tokens=0,
            cache_salt=cache_salt,
        )
        blocks = self._slots_to_blocks(device_indices[start:aligned_end])
        key = self._create_key(
            lookup, start=start, end=aligned_end, worker_id=self.worker_id
        )
        try:
            event = self._new_event()
            future = self._transfer_ctx.submit_store(
                request_id,
                key,
                self.instance_id,
                self._kv_caches,
                [blocks],
                event,
                self.blocks_in_chunk,
            )
        except Exception:
            logger.exception("LMCache store submission failed for %s", request_id)
            future = _ImmediateFuture(False)
            event = None
        if event is not None:
            future.retain_reference(event)
        self._store_submitted_tokens[request_id] = aligned_end
        return LMCacheStoreOperation(request_id, start, aligned_end, future)

    def complete_store(
        self, operation: LMCacheStoreOperation, *, synchronize: bool = True
    ) -> bool:
        if operation.result is not None:
            return operation.result
        success = False
        try:
            success = bool(operation.future.result(timeout=0))
        except Exception:
            logger.exception("LMCache store failed for %s", operation.request_id)
        operation.result = self._sync_success(success) if synchronize else success
        if (
            not operation.result
            and self._store_submitted_tokens.get(operation.request_id)
            == operation.end
        ):
            # Allow a later chunk/final store to retry this failed tail.
            self._store_submitted_tokens[operation.request_id] = operation.start
        return operation.result

    def end_lookup(self, request_id: str) -> None:
        operation = self._lookups.pop(request_id, None)
        if operation is None:
            return
        if operation.locks_held and operation.total_hit_tokens:
            self._free_lookup_locks(
                operation, operation.lock_start, operation.total_hit_tokens
            )
            operation.locks_held = False

    def end_session(self, request_id: str) -> None:
        from lmcache.v1.multiprocess.protocol import RequestType

        was_active = (
            request_id in self._active_sessions or request_id in self._lookups
        )
        self.end_lookup(request_id)
        if was_active and self.is_lookup_leader:
            self._track_control_future(
                self._send_request(
                    self._mq_client, RequestType.END_SESSION, [request_id]
                )
            )
        self._active_sessions.discard(request_id)

    def finish_request(self, request_id: str) -> None:
        self.end_session(request_id)
        self._store_submitted_tokens.pop(request_id, None)

    def end_all_sessions(self) -> None:
        for request_id in list(self._active_sessions):
            self.end_session(request_id)

    def clear(self) -> bool:
        from lmcache.v1.multiprocess.protocol import RequestType

        success = True
        try:
            if self.is_lookup_leader:
                self._send_request(self._mq_client, RequestType.CLEAR, []).result(
                    timeout=self._mq_timeout
                )
        except Exception:
            logger.exception("Failed to clear LMCache MP storage")
            success = False
        return self._sync_success(success)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from lmcache.v1.multiprocess.protocol import RequestType

        self.end_all_sessions()
        self._flush_control_futures()
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self._heartbeat_interval))
            self._heartbeat_thread = None
        if self._registered:
            try:
                self._send_request(
                    self._mq_client,
                    RequestType.UNREGISTER_KV_CACHE,
                    [self.instance_id],
                ).result(timeout=self._mq_timeout)
            except Exception:
                logger.warning("Failed to unregister LMCache KV tensors", exc_info=True)
            self._registered = False
        if self._transfer_ctx is not None:
            self._transfer_ctx.close()
            self._transfer_ctx = None
        self._mq_client.close()
