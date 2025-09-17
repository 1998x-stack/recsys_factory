from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class StageConfig:
    name: str
    component: str
    params: Dict[str, Any] = field(default_factory=dict)
    topk: Optional[int] = None

@dataclass
class PipelineConfig:
    task: str                       # "cf" | "ctr"
    dataset: Dict[str, Any]
    split: Dict[str, Any]
    # CF 用到的 stages；CTR 只有 model 一个入口
    stages: List[StageConfig] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    trainer: Dict[str, Any] = field(default_factory=dict)
    evaluator: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)  # CTR 专用

def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    task = raw.get("task", "cf")
    stages = [StageConfig(**s) for s in raw.get("stages", [])]
    return PipelineConfig(
        task=task,
        dataset=raw["dataset"],
        split=raw["split"],
        stages=stages,
        metrics=raw["metrics"],
        trainer=raw["trainer"],
        evaluator=raw.get("evaluator", {}),
        model=raw.get("model", {}),
    )
