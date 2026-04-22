from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_compute_w_loader_function():
    module_path = Path(__file__).resolve().with_name("extract_features.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_w_loader"
    )
    function_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(function_module)
    namespace = {
        "np": np,
        "torch": torch,
        "tqdm": lambda loader: loader,
        "save_hdf5": lambda *args, **kwargs: None,
        "device": torch.device("cpu"),
    }
    exec(compile(function_module, str(module_path), "exec"), namespace)
    return namespace["compute_w_loader"]


def test_compute_w_loader_verbose_uses_available_path(monkeypatch, capsys):
    compute_w_loader = _load_compute_w_loader_function()
    saved_batches = []

    def fake_save_hdf5(output_path, asset_dict, attr_dict=None, mode="w"):
        saved_batches.append((output_path, asset_dict, attr_dict, mode))

    monkeypatch.setitem(compute_w_loader.__globals__, "save_hdf5", fake_save_hdf5)

    class DummyModel:
        def __call__(self, batch):
            return torch.ones((batch.shape[0], 2))

    loader = [
        {
            "img": torch.ones((1, 3, 2, 2)),
            "coord": torch.tensor([[10, 20]]),
        }
    ]

    output_path = "features.h5"

    result = compute_w_loader(output_path, loader, DummyModel(), verbose=1)

    captured = capsys.readouterr()

    assert result == output_path
    assert "processing features.h5: total of 1 batches" in captured.out
    assert len(saved_batches) == 1
    assert saved_batches[0][0] == output_path
    np.testing.assert_array_equal(saved_batches[0][1]["coords"], np.array([[10, 20]], dtype=np.int32))

