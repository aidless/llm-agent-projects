"""
models/model_manager.py - 模型管理

支持模型注册、版本管理和多模型配置。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelVersion:
    """模型版本."""
    version: str
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """模型信息."""
    name: str
    api_endpoint: str = ""
    parameters: Optional[str] = None  # e.g., "7B", "13B", "70B"
    provider: str = "unknown"
    description: str = ""
    versions: List[ModelVersion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def latest_version(self) -> Optional[str]:
        active = [v for v in self.versions if v.is_active]
        if active:
            return max(active, key=lambda v: v.created_at).version
        return self.versions[-1].version if self.versions else None

    def add_version(self, version: str) -> None:
        """添加新版本，将旧版本标记为非活跃."""
        for v in self.versions:
            v.is_active = False
        self.versions.append(ModelVersion(version=version))
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "api_endpoint": self.api_endpoint,
            "parameters": self.parameters,
            "provider": self.provider,
            "description": self.description,
            "latest_version": self.latest_version,
            "versions": [
                {"version": v.version, "is_active": v.is_active}
                for v in self.versions
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ModelManager:
    """模型管理器.

    集中管理所有注册模型的配置和版本信息。
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(
        self,
        name: str,
        api_endpoint: str = "",
        parameters: Optional[str] = None,
        provider: str = "unknown",
        description: str = "",
        version: str = "v1",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """注册新模型."""
        if name in self._models:
            raise ValueError(f"Model '{name}' already exists. Use update() instead.")

        model = ModelInfo(
            name=name,
            api_endpoint=api_endpoint,
            parameters=parameters,
            provider=provider,
            description=description,
            metadata=metadata or {},
        )
        model.add_version(version)
        self._models[name] = model
        return model

    def unregister(self, name: str) -> bool:
        """取消注册模型."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def get(self, name: str) -> Optional[ModelInfo]:
        """获取模型信息."""
        return self._models.get(name)

    def update(self, name: str, **kwargs) -> Optional[ModelInfo]:
        """更新模型信息."""
        model = self._models.get(name)
        if not model:
            return None

        for key, value in kwargs.items():
            if hasattr(model, key) and key not in ("name", "versions", "created_at"):
                setattr(model, key, value)

        model.updated_at = time.time()
        return model

    def add_version(self, name: str, version: str) -> bool:
        """为模型添加新版本."""
        model = self._models.get(name)
        if not model:
            return False
        model.add_version(version)
        return True

    def list_models(self) -> List[ModelInfo]:
        """列出所有注册的模型."""
        return list(self._models.values())

    def list_names(self) -> List[str]:
        """列出所有模型名称."""
        return list(self._models.keys())

    def search(
        self,
        provider: Optional[str] = None,
        min_params: Optional[str] = None,
    ) -> List[ModelInfo]:
        """按条件搜索模型."""
        results = list(self._models.values())

        if provider:
            results = [m for m in results if m.provider == provider]

        if min_params:
            def _param_to_num(p: Optional[str]) -> float:
                if not p:
                    return 0
                p = p.upper().replace("B", "").replace("M", "")
                try:
                    val = float(p)
                    if "M" in (p.upper() if p else ""):
                        return val / 1000
                    return val
                except ValueError:
                    return 0

            min_val = _param_to_num(min_params)
            results = [
                m for m in results
                if _param_to_num(m.parameters) >= min_val
            ]

        return results

    def to_dict(self) -> Dict[str, Any]:
        """导出所有模型信息."""
        return {
            "total_models": len(self._models),
            "models": {name: m.to_dict() for name, m in self._models.items()},
        }