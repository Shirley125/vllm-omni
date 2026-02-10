# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase


class _DummyConnector:
    def __init__(self):
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _DummyTransferAdapter(OmniTransferAdapterBase):
    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def _poll_single_request(self, *args, **kwargs):
        return None

    def _send_single_request(self, *args, **kwargs):
        return None

    def load_async(self, *args, **kwargs):
        return None

    def save_async(self, *args, **kwargs):
        return None

    def load(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def get_finished_requests(self):
        return set()


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_shutdown_stops_threads_and_closes_connector():
    connector = _DummyConnector()
    adapter = _DummyTransferAdapter(config={})
    adapter.connector = connector
    try:
        assert _wait_until(lambda: adapter.recv_thread.is_alive())
        assert _wait_until(lambda: adapter.save_thread.is_alive())

        adapter.shutdown()

        assert _wait_until(lambda: not adapter.recv_thread.is_alive())
        assert _wait_until(lambda: not adapter.save_thread.is_alive())
        assert connector.close_calls == 1
    finally:
        adapter.shutdown()


def test_shutdown_is_idempotent_and_supports_late_connector_close():
    connector = _DummyConnector()
    adapter = _DummyTransferAdapter(config={})
    adapter.connector = connector
    try:
        adapter.shutdown(close_connector=False)
        assert connector.close_calls == 0

        adapter.shutdown()
        assert connector.close_calls == 1
    finally:
        adapter.shutdown()
