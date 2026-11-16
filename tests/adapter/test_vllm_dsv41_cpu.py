"""Run CPU interpreter oracles independently of the HCU suite's device state."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["vllm_dsv41_mhc_cpu.py", "vllm_dsv41_indexer_cpu.py"])
def test_cpu_oracles(case):
    pytest.importorskip("vllm.models.deepseek_v41")
    path = Path(__file__).with_name("cases") / case
    env = dict(os.environ, HCU_ADAPTER_HOST="none", HIP_VISIBLE_DEVICES="",
               CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", VLLM_PLUGINS="",
               VLLM_ROCM_USE_AITER="0")
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(path)],
                   env=env, check=True)


def test_ct_int8_empty_and_nonempty_pcp_rows(monkeypatch):
    from types import SimpleNamespace
    import torch
    from hcu_adapter.bridges.vllm_dsv41.linear import (
        CompressedTensorsLinearMethod, HcuDSV41ChannelInt8LinearMethod,
    )

    method = object.__new__(HcuDSV41ChannelInt8LinearMethod)
    layer = SimpleNamespace(logical_widths=[3, 5])
    x = torch.empty(2, 0, 4, dtype=torch.bfloat16)
    out = method.apply(layer, x)
    assert out.shape == (2, 0, 8) and out.dtype == x.dtype
    sample = torch.ones(2, 4)
    bias = torch.ones(8)
    calls = []
    monkeypatch.setattr(CompressedTensorsLinearMethod, "apply",
                        lambda self, l, data, b: calls.append((l, data, b)) or sample)
    assert method.apply(layer, sample, bias) is sample
    assert calls == [(layer, sample, bias)]
