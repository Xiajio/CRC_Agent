from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_load_params_function():
    module_path = Path(__file__).resolve().with_name("create_heatmaps.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_params"
    )
    function_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(function_module)
    namespace = {"pd": pd}
    exec(compile(function_module, str(module_path), "exec"), namespace)
    return namespace["load_params"]


def test_load_params_raises_for_nan_without_entering_pdb(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["create_heatmaps.py"])
    load_params = _load_load_params_function()

    df_entry = pd.Series({"patch_size": float("nan")})
    params = {"patch_size": 256}

    with pytest.raises(ValueError, match=r"Encountered NaN.*patch_size"):
        load_params(df_entry, params)
