from __future__ import annotations
from typing import Callable, Dict, Any

_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "dataset": {},
    "feature": {},
    "model": {},
    "stage": {},
    "metric": {},
    "trainer": {},
    "evaluator": {},
    "ann": {},
}

def register(kind: str, name: str):
    def deco(cls_or_fn):
        if name in _REGISTRY.setdefault(kind, {}):
            raise KeyError(f"{kind}:{name} already registered")
        _REGISTRY[kind][name] = cls_or_fn
        return cls_or_fn
    return deco

def get(kind: str, name: str):
    try:
        return _REGISTRY[kind][name]
    except KeyError as e:
        raise KeyError(f"Unregistered {kind}:{name}") from e

def list_registered(kind: str):
    return sorted(_REGISTRY.get(kind, {}).keys())
