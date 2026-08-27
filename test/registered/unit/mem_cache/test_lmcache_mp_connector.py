"""Small, server-free tests for the LMCache MP connector contracts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.connectors.lmcache.mp_connector import (
    LMCacheLoadOperation,
    LMCacheMPConnector,
)


class _Future:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self):
        return self.ready


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


if __name__ == "__main__":
    unittest.main()
