"""Small, server-free tests for the LMCache MP connector contracts."""

from array import array

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    LMCacheKVGroup,
    LMCacheLoadOperation,
    LMCacheLookupOperation,
    UnifiedLMCacheMPConnector,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.lmcache_unified_radix_cache import (
    LMCacheUnifiedRadixCache,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType


class _Future:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self):
        return self.ready

    def retain_reference(self, value):
        self.value = value


class _TransferContext:
    def __init__(self):
        self.store_args = None
        self.retrieve_args = None

    def submit_store(self, *args):
        self.store_args = args
        return _Future(True)

    def submit_retrieve(self, *args, **kwargs):
        self.retrieve_args = (args, kwargs)
        return _Future(True)


class _IPCCacheServerKey:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TreeCore:
    is_eagle = False

    @staticmethod
    def prefetch_anchor_info(_node_id):
        return None, None


class _Connector:
    @staticmethod
    def submit_lookup(*args, **kwargs):
        return object()


class TestUnifiedLMCacheMPConnector(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(UnifiedLMCacheMPConnector)
        self.connector.page_size = 4

    def test_slots_to_blocks_accepts_noncontiguous_pages(self):
        slots = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
        self.assertEqual(self.connector._slots_to_blocks(slots), [1, 3])

    def test_slots_to_blocks_rejects_partial_page(self):
        with self.assertRaisesRegex(ValueError, "complete SGLang pages"):
            self.connector._slots_to_blocks(torch.tensor([4, 5, 6]))

    def test_slots_to_blocks_rejects_unaligned_page(self):
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            self.connector._slots_to_blocks(torch.tensor([5, 6, 7, 8]))

    def test_slots_to_blocks_accepts_explicit_dummy_page_for_load(self):
        slots = torch.tensor([0, 0, 0, 0, 8, 9, 10, 11])
        self.assertEqual(
            self.connector._slots_to_blocks(slots, allow_dummy_page=True), [0, 2]
        )

    def test_component_block_ids_expand_to_kernel_groups(self):
        self.connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        self.connector._kernel_group_to_engine_group = (0, 1, 1)
        block_ids = self.connector._block_ids_for_transfer(
            [
                torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]),
                torch.tensor([8, 9, 10, 11, 16, 17, 18, 19]),
            ],
            allow_dummy_page=False,
        )
        self.assertEqual(block_ids, [[1, 3], [2, 4], [2, 4]])

    def test_wire_view_folds_attention_page_into_one_opaque_row(self):
        tensor = torch.arange(4 * 2 * 3).reshape(4, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=4)

        self.assertEqual(tuple(wire.shape), (1, 1, 24))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())
        self.assertTrue(torch.equal(wire.reshape(-1), tensor.reshape(-1)))

    def test_wire_view_keeps_one_mamba_state_slot_per_block(self):
        tensor = torch.arange(5 * 2 * 3).reshape(5, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=1)

        self.assertEqual(tuple(wire.shape), (5, 1, 6))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())

    def test_group_info_specs_preserve_component_address_spaces(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector._kv_groups = (
            LMCacheKVGroup(
                "full",
                (
                    torch.empty(20, 1, 8),
                    torch.empty(20, 1, 8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
            ),
            LMCacheKVGroup(
                "swa",
                (
                    torch.empty(12, 1, 8),
                    torch.empty(12, 1, 16),
                ),
                sliding_window_size=8,
                tokens_per_block=4,
                slots_per_block=4,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 1, 1))
        self.assertEqual(
            [spec["layer_indices"] for spec in specs], [(0, 1), (2,), (3,)]
        )
        self.assertEqual([spec["sw_size_tokens"] for spec in specs], [-1, 8, 8])

    def test_group_info_specs_mark_mamba_as_recurrent_one_block_window(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._kv_groups = (
            LMCacheKVGroup(
                "mamba",
                (torch.empty(12, 1, 32),),
                sliding_window_size=256,
                tokens_per_block=256,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0,))
        self.assertEqual(specs[0]["tokens_per_block"], 256)
        self.assertEqual(specs[0]["sw_size_tokens"], 256)
        self.assertTrue(specs[0]["recurrent_state"])

    def test_group_info_specs_keep_dsa_sidecar_in_full_address_space(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._kv_groups = (
            LMCacheKVGroup(
                "full",
                (
                    torch.empty(3, 1, 64, dtype=torch.bfloat16),
                    torch.empty(3, 1, 528, dtype=torch.uint8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
                tensor_rows_per_block=(1, 1),
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 0))
        self.assertEqual([spec["layer_indices"] for spec in specs], [(0,), (1,)])
        self.assertEqual([spec["engine_group_id"] for spec in specs], [0, 0])

    def test_submit_store_passes_list_of_block_ids_per_group(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
                torch.tensor([12, 13, 14, 15, 20, 21, 22, 23]),
            ],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [[1, 2], [3, 5]],
        )

    def test_submit_store_defers_unmapped_swa_page(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector.chunk_size = 8
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
                torch.tensor([0, 0, 0, 0, 12, 13, 14, 15]),
            ],
            cache_salt="",
        )

        self.assertIsNone(operation)
        self.assertNotIn("request", connector._store_submitted_tokens)

    def test_submit_store_accepts_dummy_mamba_blocks(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=1, slots_per_block=1),
            LMCacheKVGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [torch.arange(1, 9), torch.tensor([0, 7])],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [list(range(1, 9)), [0, 7]],
        )

    def test_submit_load_uses_compressed_mamba_block_ids(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=1, slots_per_block=1),
            LMCacheKVGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()
        connector._free_lookup_locks = lambda *args, **kwargs: None
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=3,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )

        operation = connector.submit_load(
            lookup,
            [torch.arange(4, 9), torch.tensor([0, 7])],
            local_hit_tokens=3,
        )

        args, kwargs = connector._transfer_ctx.retrieve_args
        self.assertEqual(args[4], [[0, 0, 0, 4, 5, 6, 7, 8], [0, 7]])
        self.assertEqual(kwargs["skip_first_n_tokens"], 3)
        self.assertEqual(operation.start, 0)
        self.assertEqual(operation.end, 8)

    def test_completed_operation_does_not_requery_future(self):
        operation = object.__new__(LMCacheLoadOperation)
        operation.result = False
        operation.future = _Future(ready=True)
        self.assertTrue(operation.query())

    def test_create_key_declares_single_kv_reader(self):
        self.connector.model_name = "model"
        self.connector.world_size = 2
        operation = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="salt",
        )
        custom_types = ModuleType("lmcache.v1.multiprocess.custom_types")
        custom_types.IPCCacheServerKey = _IPCCacheServerKey

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.multiprocess.custom_types": custom_types},
        ):
            key = self.connector._create_key(
                operation, start=0, end=4, worker_id=None
            )

        self.assertEqual(key.num_kv_readers, 1)

    def test_external_prefetch_key_matches_array_backed_tree_key(self):
        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache._external_flows = {}
        cache.tree_core = _TreeCore()
        cache.lmcache_connector = _Connector()
        cache.page_size = 4

        cache.prefetch_from_storage(
            "request",
            None,
            [3, 4],
            matched_prefix_tokens=[1, 2],
        )

        flow_key = cache._external_flows["request"].key
        self.assertIsInstance(flow_key.token_ids, array)
        tree_key = RadixKey(array("q", [1, 2, 3, 4]))
        self.assertEqual(tree_key.match(flow_key), 4)

    def test_resolve_registered_groups_maps_full_and_swa_subpools(self):
        class _Pool:
            kv_cache_layout = "nhd"

            def __init__(self, width):
                self.k_buffer = [torch.empty(12, 2, width)]
                self.v_buffer = [torch.empty(12, 2, width)]

        class _CompositePool:
            full_kv_pool = _Pool(8)
            swa_kv_pool = _Pool(4)

        class _Allocator:
            @staticmethod
            def get_kvcache():
                return _CompositePool()

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.tree_components = (ComponentType.FULL, ComponentType.SWA)
        cache._sliding_window_size = 6
        cache.page_size = 4

        groups = cache._resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "swa"])
        self.assertEqual([len(group.kv_tensors) for group in groups], [2, 2])
        self.assertEqual([group.sliding_window_size for group in groups], [-1, 8])

    def test_resolve_registered_groups_adds_dsa_indexer_as_full_sidecar(self):
        class _DSAPool:
            use_dsa = True
            kv_buffer = [torch.empty(12, 1, 8)]
            index_k_with_scale_buffer = [
                torch.empty(3, 4 * 132, dtype=torch.uint8),
                torch.empty(0, 4 * 132, dtype=torch.uint8),
            ]

        class _Allocator:
            @staticmethod
            def get_kvcache():
                return _DSAPool()

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.tree_components = (ComponentType.FULL,)
        cache.page_size = 4

        groups = cache._resolve_registered_groups()

        self.assertEqual(len(groups), 1)
        full = groups[0]
        self.assertEqual(full.name, "full")
        self.assertEqual(len(full.kv_tensors), 2)
        self.assertEqual(full.tensor_rows_per_block, (4, 1))
        self.assertEqual(full.tokens_per_block, 4)
        self.assertEqual(full.slots_per_block, 4)

    def test_resolve_registered_groups_maps_deepseek_v4_sidecars(self):
        page_count = 3
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty(page_count, 37440, dtype=torch.uint8)]
        )
        pool.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.empty(page_count, 8448, dtype=torch.uint8)
            ]
        )
        pool.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty(page_count, 1728, dtype=torch.uint8)]
        )
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty(page_count, 149760, dtype=torch.uint8)]
        )

        def state_pool(ratio, rows, width):
            return SimpleNamespace(
                ratio=ratio,
                ring_size=8 if ratio == 4 else 128,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.empty(rows, width, dtype=torch.float32)
                ),
            )

        pool.compress_state_pools = [
            state_pool(4, page_count * 8, 2048),
            state_pool(128, page_count * 128, 1024),
        ]
        pool.indexer_compress_state_pools = [
            state_pool(4, page_count * 8, 512),
            None,
        ]

        class _Allocator:
            @staticmethod
            def get_kvcache():
                return pool

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.tree_components = (ComponentType.FULL, ComponentType.SWA)
        cache._sliding_window_size = 128
        cache.page_size = 256

        groups = cache._resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "swa"])
        self.assertEqual([len(group.kv_tensors) for group in groups], [3, 3])
        self.assertEqual(groups[0].tensor_rows_per_block, (1, 1, 1))
        self.assertEqual(groups[1].tensor_rows_per_block, (1, 1, 1))
        self.assertEqual(groups[0].tokens_per_block, 256)
        self.assertEqual(groups[1].sliding_window_size, 256)
        self.assertEqual(
            [tuple(tensor.shape) for tensor in groups[1].kv_tensors[1:]],
            [(page_count, 8 * 2048 * 4), (page_count, 8 * 512 * 4)],
        )

    def test_deepseek_v4_rejects_unified_kv_layout(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = True

        with self.assertRaisesRegex(NotImplementedError, "unified_kv_triton"):
            LMCacheUnifiedRadixCache._resolve_dsv4_full_page_tensors(pool)

    def test_resolve_registered_groups_maps_mamba_state_pool(self):
        class _Pool:
            kv_cache_layout = "nhd"
            k_buffer = [torch.empty(12, 2, 8)]
            v_buffer = [torch.empty(12, 2, 8)]

        class _Allocator:
            @staticmethod
            def get_kvcache():
                return _Pool()

        class _MambaPool:
            num_mamba_layers = 1

            @staticmethod
            def _iter_transfer_state_tensors():
                yield "conv", torch.empty(1, 9, 2, 3), 0
                yield "temporal", torch.empty(1, 9, 4), 0

        class _ReqPool:
            mamba_pool = _MambaPool()

        class _MambaComponent:
            mamba_checkpoint_grid = 4

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.req_to_token_pool = _ReqPool()
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        cache._mamba_component = _MambaComponent()
        cache.page_size = 4

        groups = cache._resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "mamba"])
        mamba = groups[1]
        self.assertEqual(mamba.tokens_per_block, 4)
        self.assertEqual(mamba.slots_per_block, 1)
        self.assertEqual(mamba.sliding_window_size, 4)
        self.assertTrue(mamba.recurrent_state)
        self.assertEqual(
            [tuple(t.shape) for t in mamba.kv_tensors],
            [(9, 1, 6), (9, 1, 4)],
        )

    def test_device_indices_are_translated_per_component(self):
        class _Allocator:
            @staticmethod
            def translate_kv_indices_for_transfer(indices):
                return indices + 100

            @staticmethod
            def translate_loc_from_full_to_swa(indices):
                return indices + 200

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache._lmcache_component_types = (ComponentType.FULL, ComponentType.SWA)

        groups = cache._device_indices_by_group(torch.tensor([4, 5]))

        self.assertTrue(torch.equal(groups[0], torch.tensor([104, 105])))
        self.assertTrue(torch.equal(groups[1], torch.tensor([204, 205])))

    def test_device_indices_compress_mamba_to_checkpoint_blocks(self):
        class _Allocator:
            @staticmethod
            def translate_kv_indices_for_transfer(indices):
                return indices

        class _ReqPool:
            @staticmethod
            def translate_mamba_indices(indices):
                return indices + 10

        class _MambaComponent:
            mamba_checkpoint_grid = 4

        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.req_to_token_pool = _ReqPool()
        cache._mamba_component = _MambaComponent()
        cache._lmcache_component_types = (
            ComponentType.FULL,
            ComponentType.MAMBA,
        )

        groups = cache._device_indices_by_group(
            torch.arange(1, 9), mamba_value=torch.tensor([7])
        )

        self.assertTrue(torch.equal(groups[0], torch.arange(1, 9)))
        self.assertTrue(torch.equal(groups[1], torch.tensor([0, 17])))

    def test_external_swa_allocation_only_allocates_window_tail(self):
        class _SubAllocator:
            def __init__(self, start):
                self.start = start
                self.alloc_sizes = []

            @staticmethod
            def available_size():
                return 100

            def alloc(self, size):
                self.alloc_sizes.append(size)
                return torch.arange(self.start, self.start + size)

            def free(self, _indices):
                pass

        class _Allocator:
            def __init__(self):
                self.full_attn_allocator = _SubAllocator(4)
                self.swa_attn_allocator = _SubAllocator(100)
                self.mapping = None

            def set_full_to_swa_mapping(self, full, swa):
                self.mapping = (full, swa)

        allocator = _Allocator()
        cache = object.__new__(LMCacheUnifiedRadixCache)
        cache.token_to_kv_pool_allocator = allocator
        cache.is_swa_enabled = True
        cache._sliding_window_size = 6
        cache.page_size = 4

        full = cache._allocate_external_slots(12)

        self.assertEqual(allocator.full_attn_allocator.alloc_sizes, [12])
        self.assertEqual(allocator.swa_attn_allocator.alloc_sizes, [8])
        self.assertTrue(torch.equal(full, torch.arange(4, 16)))
        self.assertTrue(torch.equal(allocator.mapping[0], torch.arange(8, 16)))


if __name__ == "__main__":
    unittest.main()
