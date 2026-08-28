"""Small, server-free tests for the LMCache MP connector contracts."""

from array import array

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.connectors.lmcache.mp_connector import (
    LMCacheLookupOperation,
    LMCacheLoadOperation,
    LMCacheMPConnector,
)
from sglang.srt.mem_cache.lmcache_unified_radix_cache import (
    LMCacheUnifiedRadixCache,
)
from sglang.srt.mem_cache.radix_cache import RadixKey


class _Future:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self):
        return self.ready


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


class TestLMCacheMPConnector(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(LMCacheMPConnector)
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
        cache.page_size = 1

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


if __name__ == "__main__":
    unittest.main()
